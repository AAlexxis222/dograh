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

const defaultFormatBase = (value: unknown): string =>
    typeof value === "object" && value !== null ? JSON.stringify(value) : String(value);

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
