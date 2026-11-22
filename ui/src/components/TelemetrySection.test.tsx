import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TelemetrySection } from "./TelemetrySection";

const mocks = vi.hoisted(() => ({
    get: vi.fn(), save: vi.fn(), remove: vi.fn(),
    toastError: vi.fn(), toastSuccess: vi.fn(),
}));
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ user: { id: "user" }, loading: false }) }));
vi.mock("@/client/sdk.gen", () => ({
    getLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsGet: mocks.get,
    saveLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsPost: mocks.save,
    deleteLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsDelete: mocks.remove,
}));
vi.mock("sonner", () => ({ toast: { success: mocks.toastSuccess, error: mocks.toastError } }));
// The Radix Switch measures itself; jsdom has no ResizeObserver.
vi.stubGlobal("ResizeObserver", class {
    observe() {}
    unobserve() {}
    disconnect() {}
});

const stored = {
    host: "https://cloud.langfuse.com", public_key: "****ab12", secret_key: "****cd34",
    project_id: "cm1", traces_public: false, configured: true,
};

beforeEach(() => {
    vi.clearAllMocks();
    mocks.get.mockResolvedValue({ data: stored });
});

describe("TelemetrySection", () => {
    it.each([
        ["privacy_knob_forbidden", "ask an XPAND operator to make this change"],
        ["privacy_target_organization_required", "pass the target organization's id as organization_id"],
    ])("shows the hint of a %s refusal on save", async (code, hint) => {
        mocks.save.mockResolvedValue({
            error: { detail: { code, where: "POST /api/v1/organizations/langfuse-credentials", reason: "r", hint } },
        });
        render(<TelemetrySection />);
        fireEvent.click(await screen.findByRole("button", { name: "Save" }));
        await waitFor(() => expect(mocks.toastError).toHaveBeenCalledWith(hint));
    });

    it("shows a string detail from the backend on save", async () => {
        mocks.save.mockResolvedValue({ error: { detail: "No organization selected" } });
        render(<TelemetrySection />);
        fireEvent.click(await screen.findByRole("button", { name: "Save" }));
        await waitFor(() => expect(mocks.toastError).toHaveBeenCalledWith("No organization selected"));
    });

    it("keeps the credentials on screen when the delete fails", async () => {
        mocks.remove.mockResolvedValue({ error: { detail: "No Langfuse credentials found" } });
        render(<TelemetrySection />);
        fireEvent.click(await screen.findByRole("button", { name: "Remove" }));
        await waitFor(() => expect(mocks.toastError).toHaveBeenCalledWith("No Langfuse credentials found"));
        expect(screen.getByLabelText("Host")).toHaveProperty("value", stored.host);
        expect(screen.getByRole("button", { name: "Remove" })).toBeTruthy();
        expect(mocks.toastSuccess).not.toHaveBeenCalled();
    });
});
