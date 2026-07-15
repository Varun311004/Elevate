import { test } from "@playwright/test";
import { USERS } from "../data/users";
import { LoginPage } from "../pages/LoginPage";
import { expectStudentDashboard } from "../helpers/navigation.helper";

test("Remember Me keeps user logged in", async ({ page }) => {
  const login = new LoginPage(page);
  await login.open();
  await login.login(USERS.student.email, USERS.student.password, true);
  await expectStudentDashboard(page);
  await page.reload();
  await expectStudentDashboard(page);
});