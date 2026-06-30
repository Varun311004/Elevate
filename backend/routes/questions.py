"""Question routes for fetching and submitting answers."""
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, g
from ..models import db, Question, AnswerLog, UserProgress, EmotionLog, SubjectPerformance, User
from ..security import require_auth, optional_auth, role_required
from ..validation import sanitize_string
from sqlalchemy import and_, func
import random

from ..ai_topic_service import generate_topic_mcqs

from ..recommendation_service import (
    get_recommender_metadata,
    recommend_questions_for_user,
)
from ..adaptive_engine import compute_adaptive_signals, difficulty_fallback_sequence

questions_bp = Blueprint("questions", __name__)


STEM_SUBJECT_ALIASES = {
    "science": ["science", "physics", "chemistry", "biology"],
    "technology": ["technology", "computer science", "programming", "coding"],
    "engineering": ["engineering"],
    "mathematics": ["mathematics", "math"]
}


def normalize_topic_token(topic_value: str | None) -> str:
    """Normalize topic values so AI Basics/ai-basics/ai_basics match consistently."""
    if not topic_value:
        return ""
    normalized = sanitize_string(topic_value).strip().lower()
    if not normalized:
        return ""
    normalized = normalized.replace("-", " ")
    normalized = "_".join(normalized.split())
    return normalized


def _select_adaptive_questions(base_filters: dict, count: int, target_difficulty: str | None, user_id: int | None, exclude_answered: bool):
    """Select questions by target difficulty with nearest fallback ordering."""
    picked = []
    seen_ids = set()

    answered_ids = set()
    if user_id:
        answered_ids = {
            row.question_id
            for row in db.session.query(AnswerLog.question_id).filter(AnswerLog.user_id == user_id).all()
        }

    def query_for(difficulty_level: str, include_answered: bool, limit: int):
        q = Question.query.filter(Question.grade == sanitize_string(base_filters["grade"]).strip().lower())
        if base_filters.get("topic"):
            q = apply_topic_filter(q, base_filters["topic"])
        if base_filters.get("subject"):
            q = apply_subject_filter(q, base_filters["subject"])
        q = q.filter(Question.difficulty == difficulty_level)

        if user_id and not include_answered:
            q = q.filter(~Question.id.in_(answered_ids))

        rows = q.order_by(func.random()).limit(limit).all()
        return rows

    ordered = [d for d in difficulty_fallback_sequence(target_difficulty) if d in {"easy", "medium", "hard"}]
    if not ordered:
        ordered = ["medium", "easy", "hard"]
    if target_difficulty == "easy":
        ratios = [0.7, 0.3, 0.0]
    elif target_difficulty == "hard":
        ratios = [0.0, 0.35, 0.65]
    else:
        ratios = [0.2, 0.6, 0.2]

    requested_per_level = {}
    remaining = count
    for idx, level in enumerate(ordered):
        planned = int(round(count * ratios[idx]))
        requested_per_level[level] = min(max(planned, 0), remaining)
        remaining -= requested_per_level[level]
    for level in ordered:
        if remaining <= 0:
            break
        requested_per_level[level] += 1
        remaining -= 1

    for difficulty in ordered:
        if len(picked) >= count:
            break
        needed = max(0, requested_per_level.get(difficulty, 0))
        if needed <= 0:
            continue
        rows = query_for(difficulty, include_answered=not exclude_answered, limit=needed * 2)
        for row in rows:
            if row.id in seen_ids:
                continue
            seen_ids.add(row.id)
            picked.append(row)
            if len([q for q in picked if q.difficulty == difficulty]) >= needed:
                break

    if len(picked) < count:
        for difficulty in ordered:
            if len(picked) >= count:
                break
            needed = count - len(picked)
            rows = query_for(difficulty, include_answered=not exclude_answered, limit=needed)
            for row in rows:
                if row.id in seen_ids:
                    continue
                seen_ids.add(row.id)
                picked.append(row)

    if exclude_answered and len(picked) < count:
        for difficulty in ordered:
            if len(picked) >= count:
                break
            needed = count - len(picked)
            rows = query_for(difficulty, include_answered=True, limit=needed)
            for row in rows:
                if row.id in seen_ids:
                    continue
                seen_ids.add(row.id)
                picked.append(row)

    return picked


def apply_subject_filter(query, subject_value: str):
    """Apply case-insensitive subject filtering with STEM alias support."""
    normalized = sanitize_string(subject_value).strip().lower()
    if not normalized:
        return query

    accepted = STEM_SUBJECT_ALIASES.get(normalized, [normalized])
    accepted = [v.lower() for v in accepted]
    return query.filter(func.lower(Question.subject).in_(accepted))


def apply_topic_filter(query, topic_value: str):
    """Apply topic filtering using normalized slugs across whitespace/hyphen variants."""
    normalized_topic = normalize_topic_token(topic_value)
    if not normalized_topic:
        return query

    db_normalized_topic = func.replace(
        func.replace(func.lower(Question.syllabus_topic), '-', '_'),
        ' ',
        '_',
    )
    return query.filter(db_normalized_topic == normalized_topic)


@questions_bp.get("/")
@optional_auth
def list_questions():
    """List and filter questions by subject, grade, and difficulty."""
    from flask import g
    current_user = g.current_user
    try:
        subject = request.args.get("subject")
        grade = request.args.get("grade")
        difficulty = request.args.get("difficulty")
        topic = request.args.get("topic") or request.args.get("syllabus_topic")
        exclude_answered = request.args.get("exclude_answered", "false").lower() == "true"
        limit = min(int(request.args.get("limit", 10)), 50)
        offset = int(request.args.get("offset", 0))
        
        query = Question.query
        
        if subject:
            query = apply_subject_filter(query, subject)
        if grade:
            query = query.filter(Question.grade == sanitize_string(grade).strip().lower())
        if difficulty:
            query = query.filter(Question.difficulty == sanitize_string(difficulty).strip().lower())
        if topic:
            query = apply_topic_filter(query, topic)
        
        if exclude_answered and current_user:
            answered_question_ids = db.session.query(AnswerLog.question_id).filter(
                AnswerLog.user_id == current_user.id
            ).distinct().subquery()
            query = query.filter(~Question.id.in_(answered_question_ids))
        
        total = query.count()
        questions = query.offset(offset).limit(limit).all()
        
        questions_list = [
            {
                "id": q.id,
                "subject": q.subject,
                "grade": q.grade,
                "difficulty": q.difficulty,
                "text": q.text,
                "options": q.options,
                "hint": q.hint,
                "syllabus_topic": q.syllabus_topic,
                "readability_level": q.readability_level,
                "tags": q.tags or []
            }
            for q in questions
        ]
        
        return jsonify({
            "questions": questions_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "subject": subject,
                "grade": grade,
                "difficulty": difficulty,
                "exclude_answered": exclude_answered
            }
        }), 200
    except ValueError as e:
        return jsonify({"error": "Invalid pagination parameters"}), 400
    except Exception as e:
        return jsonify({"error": "Failed to fetch questions", "details": str(e)}), 500


@questions_bp.get('/generate')
@optional_auth
def generate_questions():
    """Generate a question set for a student based on grade and syllabus topic."""
    try:
        grade = request.args.get('grade')
        topic = request.args.get('topic') or request.args.get('syllabus_topic')
        subject = request.args.get('subject')
        try:
            count = min(int(request.args.get('count', 10)), 50)
        except Exception:
            count = 10
        exclude_answered = request.args.get('exclude_answered', 'false').lower() == 'true'

        if not grade:
            return jsonify({'error': 'grade is required'}), 400

        normalized_grade = sanitize_string(grade).strip().lower()
        normalized_topic = normalize_topic_token(topic)

        current_user = g.get('current_user')
        requested_difficulty = sanitize_string(request.args.get('difficulty')) if request.args.get('difficulty') else None

        adaptive = None
        target_difficulty = requested_difficulty
        if current_user and subject and not target_difficulty:
            adaptive = compute_adaptive_signals(current_user.id, subject, normalized_topic or topic)
            target_difficulty = adaptive.get('recommended_difficulty')

        recommendation_bundle = None
        recommendation_scores = {}
        recommended_questions: list[Question] = []
        if current_user and subject:
            recommendation_bundle = recommend_questions_for_user(
                user_id=current_user.id,
                subject=subject,
                grade=normalized_grade,
                topic=normalized_topic or topic,
                count=count,
                difficulty_hint=target_difficulty,
                exclude_answered=exclude_answered,
            )
            if recommendation_bundle.items:
                recommended_questions = [item.question for item in recommendation_bundle.items]
                recommendation_scores = {
                    str(item.question.id): {
                        'score': round(item.score, 4),
                        'expected_gain': round(item.expected_gain, 4),
                        'predicted_success': round(item.predicted_success, 4),
                        'novelty_bonus': round(item.novelty_bonus, 4),
                        'difficulty_alignment': round(item.difficulty_alignment, 4),
                        'popularity_bonus': round(item.popularity_bonus, 4),
                    }
                    for item in recommendation_bundle.items
                }

        fallback_questions = _select_adaptive_questions(
            base_filters={
                'grade': normalized_grade,
                'subject': subject,
                'topic': normalized_topic,
            },
            count=count,
            target_difficulty=target_difficulty,
            user_id=current_user.id if current_user else None,
            exclude_answered=exclude_answered,
        )

        questions: list[Question] = []
        seen_ids = set()
        for row in recommended_questions + fallback_questions:
            if row.id in seen_ids:
                continue
            seen_ids.add(row.id)
            questions.append(row)
            if len(questions) >= count:
                break

        recommendation_context = None
        if recommendation_bundle:
            recommendation_context = {
                **(recommendation_bundle.metadata or {}),
                'selected': len(recommended_questions),
            }
            if recommendation_scores:
                recommendation_context['scores'] = recommendation_scores

        # TOP-UP LOGIC: first attempt high-quality AI generation (teacher-grade flow),
        # then fallback templates only as final continuity layer.
        if len(questions) < count and subject:
            needed = count - len(questions)
            generation_difficulty = (target_difficulty or 'medium').strip().lower()

            generated_payload = []
            service_result = generate_topic_mcqs(
                subject=sanitize_string(subject).strip().lower(),
                grade=normalized_grade,
                difficulty=generation_difficulty,
                topic=normalized_topic or topic or "General",
                count=needed,
                seed=random.randint(1, 2_147_483_647),
                generation_mode="standard",
            )
            if service_result.get("ok"):
                for item in (service_result.get("questions") or []):
                    text = sanitize_string(item.get("text") or item.get("question") or "", max_length=4000)
                    options = item.get("options") if isinstance(item.get("options"), list) else []
                    options = [sanitize_string(str(o), max_length=300) for o in options if sanitize_string(str(o), max_length=300)]
                    if text and len(options) >= 2:
                        try:
                            correct_index = int(item.get("correct_index", 0))
                        except (TypeError, ValueError):
                            correct_index = 0
                        if correct_index < 0 or correct_index >= len(options):
                            correct_index = 0
                        generated_payload.append({
                            "text": text,
                            "options": options[:4],
                            "correct_index": correct_index,
                            "explanation": sanitize_string(item.get("explanation") or "", max_length=1500),
                            "topic": sanitize_string(item.get("topic") or normalized_topic or topic or "", max_length=128) or normalized_topic,
                            "source": "topic_ai_service",
                        })

            created_questions = []
            for item in generated_payload:
                new_q = Question(
                    subject=sanitize_string(subject).strip().lower(),
                    grade=normalized_grade,
                    difficulty=generation_difficulty,
                    text=item.get("text", "Error rendering question."),
                    options=item.get("options", []),
                    correct_index=item.get("correct_index", 0),
                    explanation=item.get("explanation", ""),
                    syllabus_topic=item.get("topic", normalized_topic),
                    is_generated=True,
                    generation_meta={"source": item.get("source", "topic_ai_service")}
                )
                db.session.add(new_q)
                created_questions.append(new_q)

            db.session.commit()
            questions.extend(created_questions)

        questions = questions[:count]

        questions_list = [
            {
                'id': q.id,
                'subject': q.subject,
                'grade': q.grade,
                'difficulty': q.difficulty,
                'text': q.text,
                'options': q.options,
                'hint': q.hint,
                'syllabus_topic': q.syllabus_topic,
                'readability_level': q.readability_level,
                'tags': q.tags or []
            }
            for q in questions
        ]
        return jsonify({
            'questions': questions_list,
            'count': len(questions_list),
            'adaptive': {
                'target_difficulty': target_difficulty,
                'source': 'bkt_irt' if adaptive else ('request' if requested_difficulty else 'default'),
                **(adaptive or {}),
            },
            'recommendation': recommendation_context,
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to generate questions', 'details': str(e)}), 500


@questions_bp.get('/recommendations')
@require_auth
@role_required('student')
def recommendation_endpoint():
    try:
        current_user = g.current_user
        subject = request.args.get('subject')
        if not subject:
            return jsonify({'error': 'subject is required'}), 400

        grade = request.args.get('grade') or (current_user.grade if current_user else None)
        topic = request.args.get('topic') or request.args.get('syllabus_topic')
        requested_difficulty = request.args.get('difficulty')
        exclude_answered = request.args.get('exclude_answered', 'true').lower() != 'false'
        try:
            count = min(int(request.args.get('count', 10)), 25)
        except Exception:
            return jsonify({'error': 'count must be an integer'}), 400

        bundle = recommend_questions_for_user(
            user_id=current_user.id,
            subject=subject,
            grade=sanitize_string(grade).strip().lower() if grade else None,
            topic=topic,
            count=count,
            difficulty_hint=requested_difficulty,
            exclude_answered=exclude_answered,
        )

        recommendation_scores = {}
        recommendations_payload = []
        for item in bundle.items:
            q = item.question
            recommendations_payload.append({
                'id': q.id,
                'subject': q.subject,
                'grade': q.grade,
                'difficulty': q.difficulty,
                'text': q.text,
                'options': q.options,
                'hint': q.hint,
                'syllabus_topic': q.syllabus_topic,
                'readability_level': q.readability_level,
                'tags': q.tags or [],
            })
            recommendation_scores[str(q.id)] = {
                'score': round(item.score, 4),
                'expected_gain': round(item.expected_gain, 4),
                'predicted_success': round(item.predicted_success, 4),
                'novelty_bonus': round(item.novelty_bonus, 4),
                'difficulty_alignment': round(item.difficulty_alignment, 4),
                'popularity_bonus': round(item.popularity_bonus, 4),
            }

        meta = bundle.metadata or {}
        if recommendation_scores:
            meta = {**meta, 'scores': recommendation_scores}
        manifest_meta = get_recommender_metadata() or {}
        meta.setdefault('artifact_id', manifest_meta.get('artifact_id'))
        meta.setdefault('metrics', manifest_meta.get('metrics'))
        meta['returned'] = len(recommendations_payload)

        return jsonify({'recommendations': recommendations_payload, 'meta': meta}), 200
    except Exception as exc:
        return jsonify({'error': 'Failed to fetch recommendations', 'details': str(exc)}), 500


@questions_bp.post('/generate')
@require_auth
@role_required('teacher','admin')
def generate_and_persist():
    """Generate and persist questions (teacher/admin only)."""
    try:
        data = request.get_json() or {}
        topic = data.get('topic')
        difficulty = data.get('difficulty', 'medium')
        subject = data.get('subject', 'general')
        grade = data.get('grade')
        test_title = data.get('title')
        test_description = data.get('description')
        generation_mode = data.get('generation_mode') or 'standard'
        try:
            count = min(int(data.get('count', 5)), 50)
        except Exception:
            count = 5

        seed = data.get('seed')
        if seed is not None:
            try:
                seed = int(seed)
            except Exception:
                return jsonify({'error': 'seed must be an integer'}), 400

        # Teacher Endpoint: We safely use the local AI generation service here
        service_result = generate_topic_mcqs(
            subject=subject,
            grade=grade or 'high',
            difficulty=difficulty,
            topic=topic,
            count=count,
            seed=seed,
            test_title=test_title,
            test_description=test_description,
            generation_mode=generation_mode,
        )

        payload = []
        if service_result.get("ok"):
            payload = service_result.get("questions", [])
        
        # If the AI fails, inject the fallback
        if not service_result.get("ok") or len(payload) == 0:
            payload = generate_fallback_mcqs(
                topic=topic,
                count=count,
                difficulty=difficulty,
                subject=subject,
                grade=grade or 'high'
            )

        new_questions = []
        for item in payload:
            q = Question(
                subject=subject,
                grade=grade or 'high',
                difficulty=difficulty,
                text=item.get("text", ""),
                options=item.get("options", []),
                correct_index=item.get("correct_index", 0),
                explanation=item.get("explanation", ""),
                syllabus_topic=topic,
                is_generated=True,
                generated_by=g.current_user.id,
                generation_meta={"seed": seed, "source": item.get("source", "topic_ai_service")}
            )
            db.session.add(q)
            new_questions.append(q)

        db.session.commit()

        return jsonify({'generated': [{'id': q.id, 'text': q.text, 'options': q.options, 'correct_index': q.correct_index, 'generation_meta': q.generation_meta} for q in new_questions]}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to generate and persist questions', 'details': str(e)}), 500

@questions_bp.get('/topics')
@optional_auth
def list_topics():
    """List available syllabus topics for a given subject and/or grade."""
    try:
        subject = request.args.get('subject')
        grade = request.args.get('grade')
        q = Question.query
        if subject:
            q = apply_subject_filter(q, subject)
        if grade:
            q = q.filter(Question.grade == sanitize_string(grade).strip().lower())
        topics = q.with_entities(Question.syllabus_topic).distinct().all()
        cleaned = [t[0] for t in topics if t[0]]
        return jsonify({'topics': cleaned}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to list topics', 'details': str(e)}), 500


@questions_bp.get("/<int:question_id>")
@optional_auth
def get_question(question_id):
    """Get a single question by ID."""
    from flask import g
    current_user = g.current_user
    try:
        question = db.session.get(Question, question_id)
        if not question:
            return jsonify({"error": "Question not found"}), 404
        
        already_answered = False
        if current_user:
            answer = AnswerLog.query.filter_by(
                user_id=current_user.id,
                question_id=question_id
            ).first()
            already_answered = answer is not None
        
        response = {
            "id": question.id,
            "subject": question.subject,
            "grade": question.grade,
            "difficulty": question.difficulty,
            "text": question.text,
            "options": question.options,
            "hint": question.hint,
            "tags": question.tags or [],
            "already_answered": already_answered
        }
        
        if already_answered:
            response["explanation"] = question.explanation
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": "Failed to fetch question", "details": str(e)}), 500


@questions_bp.post("/<int:question_id>/submit")
@require_auth
def submit_answer(question_id):
    """Submit an answer to a question."""
    from flask import g
    current_user = g.current_user
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        selected_index = data.get("selected_index")
        time_spent = data.get("time_spent", 0)
        emotion = data.get("emotion")
        
        if selected_index is None:
            return jsonify({"error": "selected_index is required"}), 400
        
        try:
            selected_index = int(selected_index)
            time_spent = int(time_spent)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid data types"}), 400
        
        question = db.session.get(Question, question_id)
        if not question:
            return jsonify({"error": "Question not found"}), 404
        
        if selected_index < -1 or selected_index >= len(question.options):
            return jsonify({"error": "Invalid option index"}), 400
        
        is_correct = selected_index >= 0 and selected_index == question.correct_index
        
        answer_log = AnswerLog(
            user_id=current_user.id,
            question_id=question_id,
            selected_index=selected_index,
            is_correct=is_correct,
            time_spent=time_spent,
            difficulty_at_time=question.difficulty,
            emotion_at_time=sanitize_string(emotion) if emotion else None,
            answered_at=datetime.now(timezone.utc)
        )
        db.session.add(answer_log)
        
        if emotion:
            emotion_log = EmotionLog(
                user_id=current_user.id,
                emotion=sanitize_string(emotion),
                confidence=1.0,
                context=f"answering_{question.subject}",
                timestamp=datetime.now(timezone.utc)
            )
            db.session.add(emotion_log)
        
        progress = UserProgress.query.filter_by(
            user_id=current_user.id,
            subject=question.subject
        ).first()
        
        if not progress:
            progress = UserProgress(
                user_id=current_user.id,
                subject=question.subject,
                total_questions=0,
                correct_answers=0,
                current_difficulty=question.difficulty
            )
            db.session.add(progress)
        
        progress.total_questions = int(progress.total_questions or 0) + 1
        if is_correct:
            progress.correct_answers = int(progress.correct_answers or 0) + 1
        progress.last_updated = datetime.now(timezone.utc)
        
        accuracy = (progress.correct_answers / progress.total_questions * 100) if progress.total_questions > 0 else 0
        
        subject_perf = SubjectPerformance.query.filter_by(
            user_id=current_user.id,
            subject=question.subject
        ).first()
        
        if not subject_perf:
            subject_perf = SubjectPerformance(
                user_id=current_user.id,
                subject=question.subject,
                accuracy=0.0,
                streak=0,
                best_streak=0,
                total_time_spent=0
            )
            db.session.add(subject_perf)
        
        subject_perf.streak = int(subject_perf.streak or 0)
        subject_perf.best_streak = int(subject_perf.best_streak or 0)
        subject_perf.total_time_spent = int(subject_perf.total_time_spent or 0)

        if is_correct:
            subject_perf.streak = int(subject_perf.streak or 0) + 1
            if subject_perf.streak > subject_perf.best_streak:
                subject_perf.best_streak = subject_perf.streak
        else:
            subject_perf.streak = 0
        
        subject_perf.accuracy = accuracy
        subject_perf.total_time_spent = int(subject_perf.total_time_spent or 0) + int(time_spent or 0)
        subject_perf.last_practiced_at = datetime.now(timezone.utc)

        adaptive = compute_adaptive_signals(
            user_id=current_user.id,
            subject=question.subject,
            concept=question.syllabus_topic,
        )
        if adaptive.get('recommended_difficulty'):
            progress.current_difficulty = adaptive['recommended_difficulty']
        
        db.session.commit()
        
        return jsonify({
            "correct": is_correct,
            "correct_index": question.correct_index,
            "explanation": question.explanation,
            "progress": {
                "subject": question.subject,
                "total_questions": progress.total_questions,
                "correct_answers": progress.correct_answers,
                "accuracy": round(accuracy, 2),
                "current_difficulty": progress.current_difficulty
            },
            "adaptive": adaptive,
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to submit answer", "details": str(e)}), 500