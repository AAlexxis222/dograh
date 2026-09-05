import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
    getEffective: vi.fn(),
    getOrgDefaults: vi.fn(),
    updateWorkflow: vi.fn(),
    validateWorkflow: vi.fn(),
    push: vi.fn(),
}));

vi.mock("@/client", () => ({
    getWorkflowEffectiveConfigurationApiV1WorkflowWorkflowIdConfigurationEffectiveGet: mocks.getEffective,
    getWorkflowConfigurationEffectiveDefaultsApiV1OrganizationsWorkflowConfigurationEffectiveDefaultsGet: mocks.getOrgDefaults,
    updateWorkflowApiV1WorkflowWorkflowIdPut: mocks.updateWorkflow,
    validateWorkflowApiV1WorkflowWorkflowIdValidatePost: mocks.validateWorkflow,
    createWorkflowRunApiV1WorkflowWorkflowIdRunsPost: vi.fn(),
}));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: mocks.push }) }));
vi.mock("posthog-js", () => ({ default: { capture: vi.fn() } }));
vi.mock("sonner", () => ({ toast: { error: vi.fn(), success: vi.fn() } }));
// Stable references: the canvas init effect depends on `bySpecName`, so a
// fresh Map per render would re-initialise the store on every render.
vi.mock("@/components/flow/renderer", () => {
    const nodeSpecs = { specs: [], bySpecName: new Map(), loading: false };
    return { useNodeSpecs: () => nodeSpecs };
});

import { useWorkflowStore } from "@/app/workflow/[workflowId]/stores/workflowStore";

import { useWorkflowState } from "./useWorkflowState";

const effectiveResponse = {
    effective: { max_call_duration: 600, dictionary: "house", ambient_noise_configuration: { enabled: false, volume: 0.3 } },
    own: { dictionary: "mine" },
    base: { max_call_duration: 600, dictionary: "house", ambient_noise_configuration: { enabled: false, volume: 0.3 } },
    warnings: [],
    definition_id: 7,
    definition_status: "draft",
};
const orgDefaultsResponse = {
    workflow_configurations: effectiveResponse.base,
    default_call_dispositions: [{ code: "house_code", description: "x" }],
    text_chat_inactivity_timeout_constraints: { default_seconds: 300, minimum_seconds: 60, maximum_seconds: 3600 },
    widget_text_defaults: {},
    llm: {}, tts: {}, stt: {}, warnings: [],
};

// Stable like the page's memoised `stableUser`: `user` is a dependency of the
// validate-on-mount effect, so a fresh object per render would loop it.
const user = { id: "u1" };

const renderState = () => renderHook(() => useWorkflowState({
    initialWorkflowName: "W",
    workflowId: 42,
    user,
}));

describe("useWorkflowState configuration provenance", () => {
    beforeEach(() => {
        useWorkflowStore.getState().clearStore();
        mocks.getEffective.mockReset().mockResolvedValue({ data: effectiveResponse });
        mocks.getOrgDefaults.mockReset().mockResolvedValue({ data: orgDefaultsResponse });
        mocks.updateWorkflow.mockReset().mockResolvedValue({ data: { name: "W" } });
        mocks.validateWorkflow.mockReset().mockResolvedValue({ data: { is_valid: true, errors: [] } });
    });

    it("loads effective/own/base from the per-workflow endpoint and the catalog from the org envelope", async () => {
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
        expect(result.current.configurationState?.own).toEqual({ dictionary: "mine" });
        expect(result.current.configurationState?.effective.max_call_duration).toBe(600);
        expect(result.current.defaultCallDispositions).toEqual([{ code: "house_code", description: "x" }]);
        expect(mocks.getEffective).toHaveBeenCalledWith({ path: { workflow_id: 42 } });
    });

    it("blocks the editor when either load fails, and recovers on reload", async () => {
        mocks.getEffective.mockResolvedValueOnce({ error: { detail: "boom" } });
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationLoadError).not.toBeNull());
        expect(result.current.configurationState).toBeNull();
        await act(() => result.current.reloadConfiguration());
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
        expect(result.current.configurationLoadError).toBeNull();
    });

    it("saves own ∪ patch, never the effective document, and re-reads the layers", async () => {
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
        mocks.getEffective.mockResolvedValueOnce({
            data: { ...effectiveResponse, own: { dictionary: "mine", max_call_duration: 900 } },
        });
        await act(() => result.current.saveWorkflowConfigurations(
            { set: [{ path: ["max_call_duration"], value: 900 }], unset: [] },
        ));
        const body = mocks.updateWorkflow.mock.calls[0][0].body;
        expect(body.workflow_configurations).toEqual({ dictionary: "mine", max_call_duration: 900 });
        expect(body.workflow_configurations).not.toHaveProperty("ambient_noise_configuration");
        expect(body.name).toBe("W");
        expect(mocks.getEffective).toHaveBeenCalledTimes(2);
        expect(result.current.configurationState?.own).toEqual({ dictionary: "mine", max_call_duration: 900 });
    });

    it("returning a leaf to the base removes it from own and does not inject dictionary", async () => {
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
        await act(() => result.current.saveWorkflowConfigurations({ set: [], unset: [["dictionary"]] }));
        expect(mocks.updateWorkflow.mock.calls[0][0].body.workflow_configurations).toEqual({});
    });

    it("renames with a name-only PUT", async () => {
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
        await act(() => result.current.renameWorkflow("New"));
        const body = mocks.updateWorkflow.mock.calls[0][0].body;
        expect(body).toEqual({ name: "New", workflow_definition: null });
        expect(useWorkflowStore.getState().workflowName).toBe("New");
    });

    it("drops the previous document while a reload is in flight", async () => {
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
        let resolveLayers: (value: unknown) => void = () => {};
        mocks.getEffective.mockReturnValueOnce(new Promise((resolve) => { resolveLayers = resolve; }));
        act(() => { void result.current.reloadConfiguration(); });
        expect(result.current.configurationState).toBeNull();
        expect(result.current.configurationLoadError).toBeNull();
        await expect(result.current.saveWorkflowConfigurations({ set: [{ path: ["dictionary"], value: "x" }], unset: [] }))
            .rejects.toThrow(/not loaded/);
        await act(async () => { resolveLayers({ data: effectiveResponse }); });
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
    });

    it("ignores a stale load that resolves after a later one", async () => {
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
        let resolveStale: (value: unknown) => void = () => {};
        mocks.getEffective.mockReturnValueOnce(new Promise((resolve) => { resolveStale = resolve; }));
        mocks.getEffective.mockResolvedValueOnce({ data: { ...effectiveResponse, own: { dictionary: "new" } } });
        act(() => { void result.current.reloadConfiguration(); });
        await act(() => result.current.reloadConfiguration());
        expect(result.current.configurationState?.own).toEqual({ dictionary: "new" });
        await act(async () => { resolveStale({ data: { ...effectiveResponse, own: { dictionary: "stale" } } }); });
        expect(result.current.configurationState?.own).toEqual({ dictionary: "new" });
        expect(result.current.configurationLoadError).toBeNull();
    });

    it("rejects when the re-read after a successful PUT fails", async () => {
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationState).not.toBeNull());
        mocks.getEffective.mockResolvedValueOnce({ error: { detail: "boom" } });
        await expect(act(() => result.current.saveWorkflowConfigurations(
            { set: [{ path: ["max_call_duration"], value: 900 }], unset: [] },
        ))).rejects.toThrow(/reloading/);
        expect(mocks.updateWorkflow).toHaveBeenCalledTimes(1);
        expect(result.current.configurationState).toBeNull();
        expect(result.current.configurationLoadError).not.toBeNull();
    });

    it("refuses to save before the layers are loaded", async () => {
        mocks.getEffective.mockResolvedValueOnce({ error: { detail: "boom" } });
        const { result } = renderState();
        await waitFor(() => expect(result.current.configurationLoadError).not.toBeNull());
        await expect(result.current.saveWorkflowConfigurations({ set: [{ path: ["dictionary"], value: "x" }], unset: [] }))
            .rejects.toThrow(/not loaded/);
        expect(mocks.updateWorkflow).not.toHaveBeenCalled();
    });
});
