import { expect, Page } from "@playwright/test";

export async function expectStudentDashboard(page: Page) {
  await expect(page).toHaveURL(/dashboard\.html/);
}

export async function expectTeacherDashboard(page: Page) {
  await expect(page).toHaveURL(/teacher-dashboard\.html/);
}

export async function expectAdminDashboard(page: Page) {
  await expect(page).toHaveURL(/admin\.html/);
}