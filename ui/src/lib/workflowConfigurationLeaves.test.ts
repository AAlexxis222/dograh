import { describe, expect, it } from "vitest";

import {
    annotateInheritedLeaves,
    applyConfigurationPatch,
    buildConfigurationPatch,
    collectLeafPaths,
    deleteAtPath,
    getAtPath,
    hasPath,
    isDeepEqual,
    leafKey,
    setAtPath,
} from "./workflowConfigurationLeaves";

describe("paths", () => {
    it("reads, tests and writes nested leaves without mutating the input", () => {
        const own = { ambient_noise_configuration: { volume: 0.7 } };
        expect(getAtPath(own, ["ambient_noise_configuration", "volume"])).toBe(0.7);
        expect(getAtPath(own, ["ambient_noise_configuration", "enabled"])).toBeUndefined();
        expect(hasPath(own, ["ambient_noise_configuration", "volume"])).toBe(true);
        expect(hasPath(own, ["ambient_noise_configuration", "enabled"])).toBe(false);
        expect(hasPath(own, ["ambient_noise_configuration"])).toBe(true);

        const next = setAtPath(own, ["ambient_noise_configuration", "enabled"], true);
        expect(next).toEqual({ ambient_noise_configuration: { volume: 0.7, enabled: true } });
        expect(own).toEqual({ ambient_noise_configuration: { volume: 0.7 } });
    });

    it("treats a primitive on the way as 'no path' instead of throwing", () => {
        expect(hasPath({ dictionary: "a,b" }, ["dictionary", "x"])).toBe(false);
        expect(getAtPath({ dictionary: "a,b" }, ["dictionary", "x"])).toBeUndefined();
    });

    it("deletes a leaf and prunes parents left empty", () => {
        const own = { ambient_noise_configuration: { volume: 0.7 }, dictionary: "x" };
        expect(deleteAtPath(own, ["ambient_noise_configuration", "volume"])).toEqual({ dictionary: "x" });
        expect(deleteAtPath(own, ["missing", "leaf"])).toEqual(own);
    });

    it("collects leaf paths with arrays and primitives as leaves", () => {
        const paths = collectLeafPaths({
            a: 1,
            call_dispositions: [{ code: "x" }],
            ambient_noise_configuration: { enabled: false, volume: 0.3 },
        }).map(leafKey);
        expect(paths.sort()).toEqual([
            "a",
            "ambient_noise_configuration.enabled",
            "ambient_noise_configuration.volume",
            "call_dispositions",
        ]);
    });

    it("compares JSON-ish values deeply", () => {
        expect(isDeepEqual([{ code: "a" }], [{ code: "a" }])).toBe(true);
        expect(isDeepEqual({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(true);
        expect(isDeepEqual({ a: 1 }, { a: 1, b: undefined })).toBe(true);
        expect(isDeepEqual(0.3, 0.30000001)).toBe(false);
        expect(isDeepEqual(null, undefined)).toBe(false);
    });
});

describe("applyConfigurationPatch", () => {
    it("sets and unsets leaves on the sparse own document only", () => {
        const own = { max_call_duration: 900, ambient_noise_configuration: { volume: 0.7 } };
        const next = applyConfigurationPatch(own, {
            set: [{ path: ["dictionary"], value: "mine" }, { path: ["ambient_noise_configuration", "enabled"], value: true }],
            unset: [["max_call_duration"]],
        });
        expect(next).toEqual({ dictionary: "mine", ambient_noise_configuration: { volume: 0.7, enabled: true } });
        expect(own.max_call_duration).toBe(900);
    });

    it("keeps keys the UI never renders untouched", () => {
        const own = { service_tuning: { llm: { openai: { temperature: 0.2 } } }, max_call_duration: 900 };
        const next = applyConfigurationPatch(own, { set: [{ path: ["max_call_duration"], value: 600 }], unset: [] });
        expect(next.service_tuning).toEqual(own.service_tuning);
    });
});

describe("buildConfigurationPatch", () => {
    const effective = {
        max_call_duration: 300,
        dictionary: "",
        ambient_noise_configuration: { enabled: false, volume: 0.3, storage_key: "k" },
        call_dispositions: [{ code: "a", description: "A" }],
    };

    it("emits set only for leaves whose local value differs from the effective one", () => {
        const patch = buildConfigurationPatch(effective, [
            { path: ["max_call_duration"], value: 300 },
            { path: ["dictionary"], value: "cats" },
            { path: ["call_dispositions"], value: [{ code: "a", description: "A" }] },
        ], new Set());
        expect(patch).toEqual({ set: [{ path: ["dictionary"], value: "cats" }], unset: [] });
    });

    it("emits unset for reverted leaves even when the local value equals the effective one (D-7: revert is explicit)", () => {
        const patch = buildConfigurationPatch(effective, [
            { path: ["max_call_duration"], value: 300 },
        ], new Set([leafKey(["max_call_duration"])]));
        expect(patch).toEqual({ set: [], unset: [["max_call_duration"]] });
    });

    it("treats an undefined local value as unset only when the effective document has the leaf", () => {
        const patch = buildConfigurationPatch(effective, [
            { path: ["ambient_noise_configuration", "storage_key"], value: undefined },
            { path: ["user_turn_stop_timeout"], value: undefined },
        ], new Set());
        expect(patch).toEqual({ set: [], unset: [["ambient_noise_configuration", "storage_key"]] });
    });

    it("keeps a value equal to the base as an explicit set when the user changed it back (D-7)", () => {
        // effective shows 600 (own), user types 300 (= base): that is a deliberate own value, not a revert.
        const patch = buildConfigurationPatch({ ...effective, max_call_duration: 600 }, [
            { path: ["max_call_duration"], value: 300 },
        ], new Set());
        expect(patch).toEqual({ set: [{ path: ["max_call_duration"], value: 300 }], unset: [] });
    });
});

describe("annotateInheritedLeaves", () => {
    it("marks leaves present on one side and absent on the other as inherited", () => {
        const { left, right } = annotateInheritedLeaves(
            { max_call_duration: 600, ambient_noise_configuration: { volume: 0.7 } },
            { ambient_noise_configuration: { volume: 0.7, enabled: true } },
        );
        expect(left).toEqual({ max_call_duration: 600, ambient_noise_configuration: { volume: 0.7, enabled: "(inherited)" } });
        expect(right).toEqual({ max_call_duration: "(inherited)", ambient_noise_configuration: { volume: 0.7, enabled: true } });
    });

    it("never overwrites an existing value on a shape conflict and copes with null sides", () => {
        const { left, right } = annotateInheritedLeaves({ dictionary: "x" }, { dictionary: { nested: 1 } });
        expect(left).toEqual({ dictionary: "x" });
        expect(right).toEqual({ dictionary: { nested: 1 } });
        expect(annotateInheritedLeaves(null, { a: 1 })).toEqual({ left: { a: "(inherited)" }, right: { a: 1 } });
    });
});
