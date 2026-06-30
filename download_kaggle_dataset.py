import os
import cv2
import albumentations as A

def augment_and_multiply_dataset(input_dir, output_dir, variations_per_image=7):
    print(f"🚀 Starting Data Augmentation...")
    print(f"Reading originals from: {input_dir}")
    print(f"Saving 5000+ images to: {output_dir}\n")

    # Define our augmentation pipeline
    # We carefully choose transformations that make sense for human faces
    transform = A.Compose([
        A.HorizontalFlip(p=0.5), # Mirrors the face
        A.Rotate(limit=20, p=0.7), # Tilts the head slightly
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8), # Simulates different lighting
        A.GaussianBlur(blur_limit=3, p=0.3), # Simulates out-of-focus cameras
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3) # Simulates webcam artifacts
    ])

    categories = ["Bored", "Focused"]
    total_generated = 0

    for category in categories:
        input_category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        
        # Create the new output folders
        os.makedirs(output_category_path, exist_ok=True)

        if not os.path.exists(input_category_path):
            print(f"⚠️ Warning: Could not find folder '{input_category_path}'. Skipping.")
            continue

        images = [f for f in os.listdir(input_category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(images)} original images in '{category}'. Multiplying...")

        for img_name in images:
            img_path = os.path.join(input_category_path, img_name)
            
            # Read the original image
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # OpenCV reads in BGR, we need RGB for standard processing
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1. Save the original image to the new folder first
            base_name, ext = os.path.splitext(img_name)
            cv2.imwrite(os.path.join(output_category_path, f"{base_name}_orig{ext}"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            total_generated += 1

            # 2. Generate and save the augmented variations
            for i in range(variations_per_image):
                augmented = transform(image=image)
                aug_image = augmented['image']
                
                # Convert back to BGR to save properly with OpenCV
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                
                new_filename = f"{base_name}_aug_{i}{ext}"
                cv2.imwrite(os.path.join(output_category_path, new_filename), aug_image_bgr)
                total_generated += 1

        print(f"✅ Finished generating images for '{category}'.")

    print(f"\n🎉 Success! You now have {total_generated} training images ready to go.")

if __name__ == "__main__":
    # The folder where your 705 downloaded images currently live
    INPUT_FOLDER = "Bored_vs_Focused_Training_Data" 
    
    # The new folder where the 5,000+ images will be saved
    OUTPUT_FOLDER = "Massive_Emotion_Dataset"
    
    augment_and_multiply_dataset(INPUT_FOLDER, OUTPUT_FOLDER)