import { test } from "@playwright/test";
import { LoginPage } from "../pages/LoginPage";

test("Invalid credentials", async ({ page }) => {
  const login = new LoginPage(page);
  await login.open();
  await login.login("student@elevate.com", "wrongpassword");
  await login.expectLoginError("We could not sign you in.");
});