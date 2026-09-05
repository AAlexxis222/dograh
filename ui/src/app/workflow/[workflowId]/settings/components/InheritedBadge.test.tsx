import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { InheritedBadge } from "./InheritedBadge";

const configuration = { own: { max_call_duration: 900 }, base: { max_call_duration: 300, dictionary: "" } };

describe("InheritedBadge", () => {
    it("shows Inherited when own has no value at the path", () => {
        render(<InheritedBadge path={["dictionary"]} configuration={configuration} onRevert={vi.fn()} />);
        expect(screen.getByText("Inherited")).toBeTruthy();
        expect(screen.queryByRole("button")).toBeNull();
    });

    it("shows Custom with a revert action carrying the base value when own has the leaf", () => {
        const onRevert = vi.fn();
        render(<InheritedBadge path={["max_call_duration"]} configuration={configuration} onRevert={onRevert} />);
        expect(screen.getByText("Custom")).toBeTruthy();
        fireEvent.click(screen.getByRole("button", { name: /use organization default \(300\)/i }));
        expect(onRevert).toHaveBeenCalledWith(["max_call_duration"]);
    });

    it("shows a pending revert as inherited but unsaved", () => {
        render(<InheritedBadge path={["max_call_duration"]} configuration={configuration} reverted={new Set(["max_call_duration"])} onRevert={vi.fn()} />);
        expect(screen.getByText(/Inherited · unsaved/)).toBeTruthy();
    });

    it("renders nothing for leaves that cannot be inherited", () => {
        const { container } = render(<InheritedBadge path={["external_pbx_lead_headers"]} configuration={configuration} />);
        expect(container.innerHTML).toBe("");
    });
});
