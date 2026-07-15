import { test } from "@playwright/test";
import { USERS } from "../data/users";
import { LoginPage } from "../pages/LoginPage";

test.describe("Authentication", () => {
  test("Student Login", async ({ page }) => {
    const login = new LoginPage(page);
    await login.open();
    await login.login(USERS.student.email, USERS.student.password);
    await login.expectLoggedIn();
  });
});