"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getAtPath, hasPath, leafKey, type LeafPath, type SparseDocument } from "@/lib/workflowConfigurationLeaves";

export interface InheritedBadgeProps {
    path: LeafPath;
    // The two provenance layers of WorkflowConfigurationState, widened to the
    // sparse-document shape both of them satisfy: this component only reads the
    // leaf at `path`, so it needs neither materialised type.
    configuration: { own: SparseDocument; base: SparseDocument };
    /** Leaves of the section with an unsaved revert pending. */
    reverted?: ReadonlySet<string>;
    /** Omitted ⇒ the leaf cannot be inherited ⇒ nothing renders. */
    onRevert?: (path: LeafPath) => void;
    formatBase?: (value: unknown) => string;
}

// A long inherited value (an organization dictionary, say) would push the revert
// link off the row, so the label carries a readable head of it.
const MAX_BASE_LABEL_LENGTH = 40;

const defaultFormatBase = (value: unknown): string => {
    if (typeof value === "object" && value !== null) return JSON.stringify(value);
    const text = String(value);
    return text.length > MAX_BASE_LABEL_LENGTH ? `${text.slice(0, MAX_BASE_LABEL_LENGTH)}…` : text;
};

/**
 * Says whether a setting comes from the organization or from this workflow, and
 * offers the way back to the organization value. Provenance is "own has that
 * path" (see workflowConfigurationLeaves), not "equals the base": a workflow may
 * deliberately store the same value the organization has.
 */
export function InheritedBadge({
    path,
    configuration,
    reverted,
    onRevert,
    formatBase = defaultFormatBase,
}: InheritedBadgeProps) {
    if (!onRevert) return null;

    const pending = reverted?.has(leafKey(path)) ?? false;
    if (pending || !hasPath(configuration.own, path)) {
        return (
            <Badge variant="outline" className="font-normal text-muted-foreground">
                {pending ? "Inherited · unsaved" : "Inherited"}
            </Badge>
        );
    }

    const base = getAtPath(configuration.base, path);
    const baseLabel = base === undefined ? "" : ` (${formatBase(base)})`;
    return (
        <span className="flex items-center gap-2">
            <Badge
                variant="secondary"
                className="font-normal"
                title={`Set on this workflow; the organization default is ${base === undefined ? "unset" : formatBase(base)}.`}
            >
                Custom
            </Badge>
            <Button
                type="button"
                variant="link"
                size="sm"
                className="h-auto p-0 text-xs font-normal"
                onClick={() => onRevert(path)}
            >
                Use organization default{baseLabel}
            </Button>
        </span>
    );
}
