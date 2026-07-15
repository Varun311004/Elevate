import { test, expect } from "@playwright/test";
import { USERS } from "../data/users";
import { LoginPage } from "../pages/LoginPage";

test("Logout removes session", async ({ page }) => {
  const login = new LoginPage(page);
  await login.open();
  await login.login(USERS.student.email, USERS.student.password);

  await expect(page.locator("#profileMenuToggle")).toBeVisible();
  await page.locator("#profileMenuToggle").click();

  const logoutBtn = page.locator("#logoutBtn");
  await expect(logoutBtn).toBeVisible();
  await logoutBtn.click();

  await expect(page).toHaveURL(/index\.html|\/$/);
});