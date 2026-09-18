// Sparse workflow configuration documents: a leaf is any array or primitive
// reachable through plain objects. Provenance of a leaf = "own has that path".
// Merge semantics live in the API (cascade.py); this module never merges.

export type LeafPath = readonly string[];
export type SparseDocument = Record<string, unknown>;

export interface ConfigurationPatch {
    set: Array<{ path: LeafPath; value: unknown }>;
    unset: LeafPath[];
}

export const EMPTY_PATCH: ConfigurationPatch = { set: [], unset: [] };

export const INHERITED_PLACEHOLDER = "(inherited)";

const isRecord = (value: unknown): value is SparseDocument =>
    Boolean(value) && typeof value === "object" && !Array.isArray(value);

export const leafKey = (path: LeafPath): string => path.join(".");

export const isPatchEmpty = (patch: ConfigurationPatch): boolean =>
    patch.set.length === 0 && patch.unset.length === 0;

export const getAtPath = (doc: unknown, path: LeafPath): unknown => {
    let current: unknown = doc;
    for (const segment of path) {
        if (!isRecord(current)) return undefined;
        current = current[segment];
    }
    return current;
};

export const hasPath = (doc: unknown, path: LeafPath): boolean => {
    let current: unknown = doc;
    for (const segment of path) {
        if (!isRecord(current) || !(segment in current)) return false;
        current = current[segment];
    }
    return current !== undefined;
};

export const setAtPath = (doc: SparseDocument, path: LeafPath, value: unknown): SparseDocument => {
    if (path.length === 0) return doc;
    const [head, ...rest] = path;
    if (rest.length === 0) return { ...doc, [head]: value };
    const child = isRecord(doc[head]) ? doc[head] : {};
    return { ...doc, [head]: setAtPath(child, rest, value) };
};

const omitKey = (doc: SparseDocument, key: string): SparseDocument =>
    Object.fromEntries(Object.entries(doc).filter(([entryKey]) => entryKey !== key));

export const deleteAtPath = (doc: SparseDocument, path: LeafPath): SparseDocument => {
    if (path.length === 0 || !(path[0] in doc)) return doc;
    const [head, ...rest] = path;
    if (rest.length === 0) return omitKey(doc, head);
    const child = doc[head];
    if (!isRecord(child)) return doc;
    const nextChild = deleteAtPath(child, rest);
    if (Object.keys(nextChild).length === 0) return omitKey(doc, head);
    return { ...doc, [head]: nextChild };
};

export const collectLeafPaths = (doc: unknown, prefix: LeafPath = []): LeafPath[] => {
    if (!isRecord(doc)) return prefix.length ? [prefix] : [];
    return Object.entries(doc).flatMap(([key, value]) =>
        isRecord(value) ? collectLeafPaths(value, [...prefix, key]) : [[...prefix, key]],
    );
};

export const isDeepEqual = (a: unknown, b: unknown): boolean => {
    if (a === b) return true;
    if (Array.isArray(a) || Array.isArray(b)) {
        return Array.isArray(a) && Array.isArray(b) && a.length === b.length
            && a.every((item, index) => isDeepEqual(item, b[index]));
    }
    if (isRecord(a) && isRecord(b)) {
        const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
        return [...keys].every((key) => isDeepEqual(a[key], b[key]));
    }
    return false;
};

export const applyConfigurationPatch = (own: SparseDocument, patch: ConfigurationPatch): SparseDocument => {
    let next = own;
    for (const { path, value } of patch.set) next = setAtPath(next, path, value);
    for (const path of patch.unset) next = deleteAtPath(next, path);
    return next;
};

/**
 * What a settings section sends: only leaves the user changed (`set`) and
 * leaves explicitly returned to the base (`unset`). Untouched leaves stay
 * whatever they were in `own`, so an own value equal to the base remains own
 * (D-7) and an inherited one stays inherited.
 *
 * An emptied control only produces an `unset` when the workflow actually stores
 * that leaf: unsetting an inherited leaf is a no-op on the server, so emitting
 * it would leave the section permanently dirty and re-PUT an unchanged document.
 */
export const buildConfigurationPatch = (
    effective: unknown,
    own: unknown,
    leaves: Array<{ path: LeafPath; value: unknown }>,
    reverted: ReadonlySet<string>,
): ConfigurationPatch => {
    const patch: ConfigurationPatch = { set: [], unset: [] };
    for (const { path, value } of leaves) {
        if (reverted.has(leafKey(path))) {
            patch.unset.push(path);
        } else if (value === undefined) {
            if (hasPath(own, path)) patch.unset.push(path);
        } else if (!isDeepEqual(value, getAtPath(effective, path))) {
            patch.set.push({ path, value });
        }
    }
    return patch;
};

/**
 * For the version diff: a leaf stored on one side and absent on the other was
 * inherited there, not deleted. Write a placeholder so the diff says so.
 */
export const annotateInheritedLeaves = (
    left: unknown,
    right: unknown,
    placeholder: string = INHERITED_PLACEHOLDER,
): { left: unknown; right: unknown } => {
    const leftDoc: SparseDocument = isRecord(left) ? left : {};
    const rightDoc: SparseDocument = isRecord(right) ? right : {};
    const fill = (target: SparseDocument, source: SparseDocument): SparseDocument =>
        collectLeafPaths(source).reduce((doc, path) => {
            if (hasPath(doc, path)) return doc;
            // Shape conflict: an ancestor already holds a leaf here; leave it.
            const conflict = path.slice(0, -1).some((_, index) => {
                const ancestor = getAtPath(doc, path.slice(0, index + 1));
                return ancestor !== undefined && !isRecord(ancestor);
            });
            return conflict ? doc : setAtPath(doc, path, placeholder);
        }, target);
    return { left: fill(leftDoc, rightDoc), right: fill(rightDoc, leftDoc) };
};
