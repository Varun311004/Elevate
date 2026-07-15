import { expect, Locator, Page } from "@playwright/test";

export class LoginPage {
  constructor(private readonly page: Page) {}

  readonly email: Locator = this.page.locator("#loginEmail");
  readonly password: Locator = this.page.locator("#loginPassword");
  readonly rememberMe: Locator = this.page.locator("#rememberMe");
  readonly loginForm: Locator = this.page.locator("#loginForm");
  readonly loginButton: Locator = this.page.locator("form#loginForm button[type='submit']");
  readonly loginAlert: Locator = this.page.locator("#loginAlert");
  readonly showSignupLink: Locator = this.page.locator("#showSignup");

  async open() {
    await this.page.goto("/");
    await expect(this.loginForm).toBeVisible();
  }

  async login(email: string, password: string, remember = false) {
    await expect(this.email).toBeVisible();
    await this.email.fill(email);
    await this.password.fill(password);

    if (remember) {
      const checked = await this.rememberMe.isChecked().catch(() => false);
      if (!checked) {
        await this.rememberMe.check();
      }
    }

    await this.loginButton.click();
  }

  async expectLoggedIn() {
    await expect(this.page).toHaveURL(/dashboard\.html|teacher-dashboard\.html|admin\.html/);
  }

  async expectLoginError(text: string) {
    await expect(this.loginAlert).toBeVisible();
    await expect(this.loginAlert).toContainText(text);
  }

  async openSignupForm() {
    await this.showSignupLink.click();
    await expect(this.page.locator("#signupForm")).toBeVisible();
  }
}