import { Page } from "@playwright/test";

import { LoginPage } from "../pages/LoginPage";

export async function loginAs(
  page: Page,
  email: string,
  password: string
) {
  const login = new LoginPage(page);

  await login.open();

  await login.login(email, password);

  await login.expectLoggedIn();
}