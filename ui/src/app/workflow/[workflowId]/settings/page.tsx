"use client";

import { format } from "date-fns";
import { ArrowLeft, BookA, Brain, CalendarIcon, Clipboard, Download, ExternalLink, FileDown, Fingerprint, Loader2, Mic, Pause, PhoneOff, Play, Plus, Rocket, Settings, Trash2Icon, Upload, Variable, X } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import {
    downloadWorkflowReportApiV1WorkflowWorkflowIdReportGet,
    getAmbientNoiseUploadUrlApiV1WorkflowAmbientNoiseUploadUrlPost,
    getModelConfigurationV2ApiV1OrganizationsModelConfigurationsV2Get,
    getModelConfigurationV2DefaultsApiV1OrganizationsModelConfigurationsV2DefaultsGet,
    getWorkflowApiV1WorkflowFetchWorkflowIdGet,
} from "@/client/sdk.gen";
import type {
    ModelConfigurationPricingResponse,
    OrganizationAiModelConfigurationResponse,
    OrganizationAiModelConfigurationV2,
    WorkflowResponse,
} from "@/client/types.gen";
import {
    AIModelConfigurationV2Editor,
    type ModelConfigurationDefaultsV2,
} from "@/components/AIModelConfigurationV2Editor";
import { FlowEdge, FlowNode } from "@/components/flow/types";
import { LLMConfigSelector } from "@/components/LLMConfigSelector";
import SpinLoader from "@/components/SpinLoader";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { SETTINGS_DOCUMENTATION_URLS } from "@/constants/documentation";
import { useOrgConfig } from "@/context/OrgConfigContext";
import { UnsavedChangesProvider, useUnsavedChanges, useUnsavedChangesContext } from "@/context/UnsavedChangesContext";
import { useAudioPlayback } from "@/hooks/useAudioPlayback";
import { detailFromError } from "@/lib/apiError";
import { useAuth } from "@/lib/auth";
import { copyTextToClipboard } from "@/lib/clipboard";
import logger from "@/lib/logger";
import { fetchModelConfigurationPricing } from "@/lib/modelConfigurationPricing";
import {
    buildConfigurationPatch,
    type ConfigurationPatch,
    getAtPath,
    isPatchEmpty,
    leafKey,
    type LeafPath,
} from "@/lib/workflowConfigurationLeaves";
import {
    type AmbientNoiseConfiguration,
    type CallDispositionOption,
    DEFAULT_PROVISIONAL_VAD_PAUSE_SECS,
    DEFAULT_TURN_START_MIN_WORDS,
    DEFAULT_VOICEMAIL_DETECTION_CONFIGURATION,
    type ExternalPBXFieldMapping,
    TURN_START_STRATEGY_OPTIONS,
    type TurnStartStrategy,
    type TurnStopStrategy,
    type VoicemailDetectionConfiguration,
    type WorkflowConfigurationState,
} from "@/types/workflow-configurations";

import { EmbedDialog } from "../components/EmbedDialog";
import { useWorkflowState } from "../hooks/useWorkflowState";
import {
    CallDispositionEditor,
    type CallDispositionRow,
    createCallDispositionRows,
    normalizeCallDispositions,
    validateCallDispositionRows,
} from "./components/CallDispositionEditor";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PUBLISH_WORKFLOW_REMINDER = "Publish the agent to apply the changes.";

const DEFAULT_VOICEMAIL_SYSTEM_PROMPT = `You are a voicemail detection classifier for an OUTBOUND calling system. A bot has called a phone number and you need to determine if a human answered or if the call went to voicemail based on the provided text.

HUMAN ANSWERED - LIVE CONVERSATION (respond "CONVERSATION"):
- Personal greetings: "Hello?", "Hi", "Yeah?", "John speaking"
- Interactive responses: "Who is this?", "What do you want?", "Can I help you?"
- Conversational tone expecting back-and-forth dialogue
- Questions directed at the caller: "Hello? Anyone there?"
- Informal responses: "Yep", "What's up?", "Speaking"
- Natural, spontaneous speech patterns
- Immediate acknowledgment of the call

VOICEMAIL SYSTEM (respond "VOICEMAIL"):
- Automated voicemail greetings: "Hi, you've reached [name], please leave a message"
- Phone carrier messages: "The number you have dialed is not in service", "Please leave a message", "All circuits are busy"
- Professional voicemail: "This is [name], I'm not available right now"
- Instructions about leaving messages: "leave a message", "leave your name and number"
- References to callback or messaging: "call me back", "I'll get back to you"
- Carrier system messages: "mailbox is full", "has not been set up"
- Business hours messages: "our office is currently closed"

Respond with ONLY "CONVERSATION" if a person answered, or "VOICEMAIL" if it's voicemail/recording.`;

// Sidebar navigation items
const NAV_ITEMS = [
    { id: "general", label: "General", icon: Settings },
    { id: "models", label: "Model Overrides", icon: Brain },
    { id: "variables", label: "Template Variables", icon: Variable },
    { id: "dictionary", label: "Dictionary", icon: BookA },
    { id: "voicemail", label: "Voicemail Detection", icon: PhoneOff },
    { id: "recordings", label: "Recordings", icon: Mic },
    { id: "deployment", label: "Add to Website", icon: Rocket },
    { id: "report", label: "Report", icon: FileDown },
    { id: "identity", label: "Agent UUID", icon: Fingerprint },
];

// ---------------------------------------------------------------------------
// Section: Report
// ---------------------------------------------------------------------------

function ReportSection({ workflowId }: { workflowId: number }) {
    const [startDate, setStartDate] = useState<Date | undefined>(undefined);
    const [startTime, setStartTime] = useState("00:00");
    const [endDate, setEndDate] = useState<Date | undefined>(undefined);
    const [endTime, setEndTime] = useState("23:59");
    const [isPopoverOpen, setIsPopoverOpen] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);

    const buildDateTime = (date: Date | undefined, time: string): string | undefined => {
        if (!date) return undefined;
        const [hours, minutes] = time.split(":").map(Number);
        const combined = new Date(date);
        combined.setHours(hours, minutes, 0, 0);
        return combined.toISOString();
    };

    const handleDownload = async () => {
        setIsDownloading(true);
        setIsPopoverOpen(false);
        try {
            const response = await downloadWorkflowReportApiV1WorkflowWorkflowIdReportGet({
                path: { workflow_id: workflowId },
                query: {
                    start_date: buildDateTime(startDate, startTime),
                    end_date: buildDateTime(endDate, endTime),
                },
                parseAs: "blob",
            });

            if (response.data) {
                const blob = response.data as Blob;
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `workflow_${workflowId}_report.csv`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
            } else {
                toast.error("Failed to download report");
            }
        } catch (err) {
            logger.error(`Failed to download workflow report: ${err}`);
            toast.error("Failed to download report");
        } finally {
            setIsDownloading(false);
        }
    };

    const handleClear = () => {
        setStartDate(undefined);
        setStartTime("00:00");
        setEndDate(undefined);
        setEndTime("23:59");
    };

    return (
        <Card id="report">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                    <FileDown className="h-4 w-4" />
                    Report
                </CardTitle>
                <CardDescription>
                    Download a CSV report of completed runs for this agent, optionally filtered by date range.
                </CardDescription>
            </CardHeader>
            <CardFooter className="border-t pt-6">
                <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
                    <PopoverTrigger asChild>
                        <Button variant="outline" disabled={isDownloading}>
                            <Download className="h-4 w-4 mr-2" />
                            Download Report
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-4" align="start">
                        <div className="space-y-4">
                            <div className="text-sm font-medium">Filter by date range</div>
                            <div className="grid gap-3">
                                <div className="space-y-1.5">
                                    <Label className="text-xs">From</Label>
                                    <div className="flex gap-2">
                                        <Popover>
                                            <PopoverTrigger asChild>
                                                <Button variant="outline" size="sm" className="w-[140px] justify-start text-left font-normal">
                                                    <CalendarIcon className="mr-2 h-3.5 w-3.5" />
                                                    {startDate ? format(startDate, "MMM dd, yyyy") : "Start date"}
                                                </Button>
                                            </PopoverTrigger>
                                            <PopoverContent className="w-auto p-0" align="start">
                                                <Calendar
                                                    mode="single"
                                                    selected={startDate}
                                                    onSelect={setStartDate}
                                                    disabled={(date) => (endDate ? date > endDate : false)}
                                                />
                                            </PopoverContent>
                                        </Popover>
                                        <Input
                                            type="time"
                                            value={startTime}
                                            onChange={(e) => setStartTime(e.target.value)}
                                            className="w-[100px] h-8 text-xs"
                                        />
                                    </div>
                                </div>
                                <div className="space-y-1.5">
                                    <Label className="text-xs">To</Label>
                                    <div className="flex gap-2">
                                        <Popover>
                                            <PopoverTrigger asChild>
                                                <Button variant="outline" size="sm" className="w-[140px] justify-start text-left font-normal">
                                                    <CalendarIcon className="mr-2 h-3.5 w-3.5" />
                                                    {endDate ? format(endDate, "MMM dd, yyyy") : "End date"}
                                                </Button>
                                            </PopoverTrigger>
                                            <PopoverContent className="w-auto p-0" align="start">
                                                <Calendar
                                                    mode="single"
                                                    selected={endDate}
                                                    onSelect={setEndDate}
                                                    disabled={(date) => (startDate ? date < startDate : false)}
                                                />
                                            </PopoverContent>
                                        </Popover>
                                        <Input
                                            type="time"
                                            value={endTime}
                                            onChange={(e) => setEndTime(e.target.value)}
                                            className="w-[100px] h-8 text-xs"
                                        />
                                    </div>
                                </div>
                            </div>
                            <Separator />
                            <div className="flex justify-between">
                                <Button variant="ghost" size="sm" onClick={handleClear}>
                                    Clear
                                </Button>
                                <Button size="sm" onClick={handleDownload} disabled={isDownloading}>
                                    <Download className="h-3.5 w-3.5 mr-1.5" />
                                    {startDate || endDate ? "Download Filtered" : "Download All"}
                                </Button>
                            </div>
                        </div>
                    </PopoverContent>
                </Popover>
            </CardFooter>
        </Card>
    );
}

// ---------------------------------------------------------------------------
// Section: General
// ---------------------------------------------------------------------------

const MAX_AMBIENT_NOISE_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// Every configuration leaf this section owns. A save carries only the ones the
// user edited (plus the ones explicitly returned to the base), never the whole
// materialised document.
const GENERAL_LEAVES = {
    ambientEnabled: ["ambient_noise_configuration", "enabled"],
    ambientVolume: ["ambient_noise_configuration", "volume"],
    ambientStorageKey: ["ambient_noise_configuration", "storage_key"],
    ambientStorageBackend: ["ambient_noise_configuration", "storage_backend"],
    ambientOriginalFilename: ["ambient_noise_configuration", "original_filename"],
    maxCallDuration: ["max_call_duration"],
    maxUserIdleTimeout: ["max_user_idle_timeout"],
    userTurnStopTimeout: ["user_turn_stop_timeout"],
    smartTurnStopSecs: ["smart_turn_stop_secs"],
    turnStartStrategy: ["turn_start_strategy"],
    turnStartMinWords: ["turn_start_min_words"],
    provisionalVadPauseSecs: ["provisional_vad_pause_secs"],
    turnStopStrategy: ["turn_stop_strategy"],
    contextCompactionEnabled: ["context_compaction_enabled"],
    callDispositions: ["call_dispositions"],
    transcriptEndTimestamps: ["transcript_configuration", "include_end_timestamps"],
    externalPbxFieldMappings: ["external_pbx_field_mappings"],
    externalPbxLeadHeaders: ["external_pbx_lead_headers"],
} as const satisfies Record<string, LeafPath>;

function GeneralSection({
    configuration,
    defaultCallDispositions,
    workflowName,
    workflowId,
    onSave,
}: {
    configuration: WorkflowConfigurationState;
    defaultCallDispositions: CallDispositionOption[];
    workflowName: string;
    workflowId: number;
    onSave: (patch: ConfigurationPatch, workflowName?: string) => Promise<void>;
}) {
    const { externalPbxIntegrationsEnabled } = useOrgConfig();
    const { effective, base } = configuration;
    const [name, setName] = useState(workflowName);
    const [ambientNoiseConfig, setAmbientNoiseConfig] = useState<AmbientNoiseConfiguration>(
        effective.ambient_noise_configuration,
    );
    const [maxCallDuration, setMaxCallDuration] = useState(effective.max_call_duration);
    const [maxUserIdleTimeout, setMaxUserIdleTimeout] = useState(effective.max_user_idle_timeout);
    const [userTurnStopTimeout, setUserTurnStopTimeout] = useState<number | undefined>(
        effective.user_turn_stop_timeout as number | undefined,
    );
    const [smartTurnStopSecs, setSmartTurnStopSecs] = useState(effective.smart_turn_stop_secs);
    const [turnStartStrategy, setTurnStartStrategy] = useState<TurnStartStrategy>(
        effective.turn_start_strategy,
    );
    const [turnStartMinWords, setTurnStartMinWords] = useState(
        effective.turn_start_min_words,
    );
    const [provisionalVadPauseSecs, setProvisionalVadPauseSecs] = useState(
        effective.provisional_vad_pause_secs,
    );
    const [turnStopStrategy, setTurnStopStrategy] = useState<TurnStopStrategy>(
        effective.turn_stop_strategy,
    );
    const [contextCompactionEnabled, setContextCompactionEnabled] = useState(
        effective.context_compaction_enabled,
    );
    const [callDispositionRows, setCallDispositionRows] = useState<CallDispositionRow[]>(
        () => createCallDispositionRows(effective.call_dispositions),
    );
    const [includeTranscriptEndTimestamps, setIncludeTranscriptEndTimestamps] = useState(
        effective.transcript_configuration?.include_end_timestamps ?? false,
    );
    const [externalPbxFieldMappings, setExternalPbxFieldMappings] = useState<ExternalPBXFieldMapping[]>(
        effective.external_pbx_field_mappings,
    );
    const [externalPbxLeadHeaders, setExternalPbxLeadHeaders] = useState<string[]>(
        effective.external_pbx_lead_headers,
    );
    // Leaves the user asked to return to the base: they travel as `unset`, so
    // the workflow stops storing them and follows the organization again.
    const [reverted, setReverted] = useState<Set<string>>(() => new Set());
    const [isSaving, setIsSaving] = useState(false);
    const [isUploadingAudio, setIsUploadingAudio] = useState(false);
    const [audioUploadError, setAudioUploadError] = useState<string | null>(null);
    const ambientFileInputRef = useRef<HTMLInputElement>(null);
    const { playingId, toggle: togglePlayback } = useAudioPlayback();
    const selectedTurnStartStrategy = TURN_START_STRATEGY_OPTIONS.find(
        (option) => option.value === turnStartStrategy,
    );
    const externalPbxFieldMappingsValid = externalPbxFieldMappings.every(
        (mapping) =>
            Boolean(mapping.context_path.trim()) &&
            /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(mapping.destination_field.trim()),
    );
    const externalPbxLeadHeadersValid = externalPbxLeadHeaders.every((field) =>
        /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(field.trim()),
    );
    const externalPbxSettingsValid =
        externalPbxFieldMappingsValid && externalPbxLeadHeadersValid;
    const normalizedCallDispositions = useMemo(
        () => normalizeCallDispositions(callDispositionRows),
        [callDispositionRows],
    );
    const callDispositionsValid = useMemo(
        () => validateCallDispositionRows(callDispositionRows).isValid,
        [callDispositionRows],
    );

    // Put one leaf back on the base value and mark it for `unset` on save. It
    // lives here because it owns the section's setters; the controls that call
    // it (the per-leaf "inherited" badges) land in the next change.
    // eslint-disable-next-line @typescript-eslint/no-unused-vars, unused-imports/no-unused-vars
    const revertLeaf = (path: LeafPath) => {
        const key = leafKey(path);
        const value = getAtPath(base, path);
        switch (key) {
            case leafKey(GENERAL_LEAVES.maxCallDuration): setMaxCallDuration(value as number); break;
            case leafKey(GENERAL_LEAVES.maxUserIdleTimeout): setMaxUserIdleTimeout(value as number); break;
            case leafKey(GENERAL_LEAVES.userTurnStopTimeout): setUserTurnStopTimeout(value as number | undefined); break;
            case leafKey(GENERAL_LEAVES.smartTurnStopSecs): setSmartTurnStopSecs(value as number); break;
            case leafKey(GENERAL_LEAVES.turnStartStrategy): setTurnStartStrategy(value as TurnStartStrategy); break;
            case leafKey(GENERAL_LEAVES.turnStartMinWords): setTurnStartMinWords(value as number); break;
            case leafKey(GENERAL_LEAVES.provisionalVadPauseSecs): setProvisionalVadPauseSecs(value as number); break;
            case leafKey(GENERAL_LEAVES.turnStopStrategy): setTurnStopStrategy(value as TurnStopStrategy); break;
            case leafKey(GENERAL_LEAVES.contextCompactionEnabled): setContextCompactionEnabled(Boolean(value)); break;
            case leafKey(GENERAL_LEAVES.transcriptEndTimestamps): setIncludeTranscriptEndTimestamps(Boolean(value)); break;
            case leafKey(GENERAL_LEAVES.callDispositions): setCallDispositionRows(createCallDispositionRows((value as CallDispositionOption[]) ?? [])); break;
            case leafKey(GENERAL_LEAVES.ambientEnabled): setAmbientNoiseConfig((prev) => ({ ...prev, enabled: Boolean(value) })); break;
            case leafKey(GENERAL_LEAVES.ambientVolume): setAmbientNoiseConfig((prev) => ({ ...prev, volume: value as number })); break;
            case leafKey(GENERAL_LEAVES.ambientStorageKey): {
                const baseAmbient = base.ambient_noise_configuration;
                setAmbientNoiseConfig((prev) => ({
                    ...prev,
                    storage_key: baseAmbient.storage_key,
                    storage_backend: baseAmbient.storage_backend,
                    original_filename: baseAmbient.original_filename,
                }));
                break;
            }
        }
        setReverted((prev) => {
            const next = new Set(prev);
            next.add(key);
            // The three custom-audio leaves travel together.
            if (key === leafKey(GENERAL_LEAVES.ambientStorageKey)) {
                next.add(leafKey(GENERAL_LEAVES.ambientStorageBackend));
                next.add(leafKey(GENERAL_LEAVES.ambientOriginalFilename));
            }
            return next;
        });
    };

    // Editing a leaf cancels a pending revert on it.
    const unrevert = (...paths: LeafPath[]) => setReverted((prev) => {
        if (!paths.some((path) => prev.has(leafKey(path)))) return prev;
        const next = new Set(prev);
        paths.forEach((path) => next.delete(leafKey(path)));
        return next;
    });

    const unrevertAmbientAudio = () => unrevert(
        GENERAL_LEAVES.ambientStorageKey,
        GENERAL_LEAVES.ambientStorageBackend,
        GENERAL_LEAVES.ambientOriginalFilename,
    );

    const leaves = useMemo(() => {
        const list: Array<{ path: LeafPath; value: unknown }> = [
            { path: GENERAL_LEAVES.ambientEnabled, value: ambientNoiseConfig.enabled },
            { path: GENERAL_LEAVES.ambientVolume, value: ambientNoiseConfig.volume },
            { path: GENERAL_LEAVES.ambientStorageKey, value: ambientNoiseConfig.storage_key },
            { path: GENERAL_LEAVES.ambientStorageBackend, value: ambientNoiseConfig.storage_backend },
            { path: GENERAL_LEAVES.ambientOriginalFilename, value: ambientNoiseConfig.original_filename },
            { path: GENERAL_LEAVES.maxCallDuration, value: maxCallDuration },
            { path: GENERAL_LEAVES.maxUserIdleTimeout, value: maxUserIdleTimeout },
            { path: GENERAL_LEAVES.userTurnStopTimeout, value: userTurnStopTimeout },
            { path: GENERAL_LEAVES.smartTurnStopSecs, value: smartTurnStopSecs },
            { path: GENERAL_LEAVES.turnStartStrategy, value: turnStartStrategy },
            { path: GENERAL_LEAVES.turnStartMinWords, value: turnStartMinWords },
            { path: GENERAL_LEAVES.provisionalVadPauseSecs, value: provisionalVadPauseSecs },
            { path: GENERAL_LEAVES.turnStopStrategy, value: turnStopStrategy },
            { path: GENERAL_LEAVES.contextCompactionEnabled, value: contextCompactionEnabled },
            { path: GENERAL_LEAVES.callDispositions, value: normalizedCallDispositions },
            { path: GENERAL_LEAVES.transcriptEndTimestamps, value: includeTranscriptEndTimestamps },
        ];
        // Not inheritable and hidden while the integration is off: absent in the
        // request means "unchanged" server-side (spec §18), so they only travel
        // when edited here.
        if (externalPbxIntegrationsEnabled) {
            list.push(
                { path: GENERAL_LEAVES.externalPbxFieldMappings, value: externalPbxFieldMappings },
                { path: GENERAL_LEAVES.externalPbxLeadHeaders, value: externalPbxLeadHeaders.map((field) => field.trim()) },
            );
        }
        return list;
    }, [
        ambientNoiseConfig,
        maxCallDuration,
        maxUserIdleTimeout,
        userTurnStopTimeout,
        smartTurnStopSecs,
        turnStartStrategy,
        turnStartMinWords,
        provisionalVadPauseSecs,
        turnStopStrategy,
        contextCompactionEnabled,
        normalizedCallDispositions,
        includeTranscriptEndTimestamps,
        externalPbxFieldMappings,
        externalPbxLeadHeaders,
        externalPbxIntegrationsEnabled,
    ]);

    const patch = useMemo(
        () => buildConfigurationPatch(effective, leaves, reverted),
        [effective, leaves, reverted],
    );
    const isDirty = name !== workflowName || !isPatchEmpty(patch);

    useUnsavedChanges("general", isDirty);

    const handleAmbientFileUpload = async (file: File) => {
        if (file.size > MAX_AMBIENT_NOISE_FILE_SIZE) {
            setAudioUploadError(`File too large (${(file.size / (1024 * 1024)).toFixed(1)}MB). Maximum is 10MB.`);
            return;
        }

        setIsUploadingAudio(true);
        setAudioUploadError(null);

        try {
            // 1. Get presigned upload URL
            const res = await getAmbientNoiseUploadUrlApiV1WorkflowAmbientNoiseUploadUrlPost({
                body: {
                    workflow_id: Number(workflowId),
                    filename: file.name,
                    mime_type: file.type || "audio/wav",
                    file_size: file.size,
                },
            });

            if (res.error || !res.data?.upload_url) {
                throw new Error("Failed to get upload URL");
            }

            const data = res.data;

            // 2. Upload file to storage
            const uploadRes = await fetch(data.upload_url, {
                method: "PUT",
                body: file,
                headers: { "Content-Type": file.type || "audio/wav" },
            });
            if (!uploadRes.ok) {
                throw new Error("File upload failed");
            }

            // 3. Update config with storage reference
            unrevertAmbientAudio();
            setAmbientNoiseConfig((prev) => ({
                ...prev,
                storage_key: data.storage_key,
                storage_backend: data.storage_backend,
                original_filename: file.name,
            }));
        } catch (err) {
            setAudioUploadError(err instanceof Error ? err.message : "Upload failed");
        } finally {
            setIsUploadingAudio(false);
            if (ambientFileInputRef.current) ambientFileInputRef.current.value = "";
        }
    };

    const handleRemoveCustomAudio = () => {
        unrevertAmbientAudio();
        setAmbientNoiseConfig((prev) => ({
            enabled: prev.enabled,
            volume: prev.volume,
        }));
    };

    const handleSave = async () => {
        setIsSaving(true);
        const callDispositionRowsAtSave = callDispositionRows;
        try {
            await onSave(patch, name);
            setReverted(new Set());
            setCallDispositionRows((current) => (
                current === callDispositionRowsAtSave
                    ? current.map((row, index) => ({
                        ...row,
                        ...normalizedCallDispositions[index],
                    }))
                    : current
            ));
            toast.success(`General settings saved. ${PUBLISH_WORKFLOW_REMINDER}`);
        } catch (error) {
            console.error("Failed to save general settings:", error);
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <Card id="general">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                    <Settings className="h-4 w-4" />
                    General
                </CardTitle>
                <CardDescription>Agent name, call behavior, and turn detection.{" "}
                    <a href={SETTINGS_DOCUMENTATION_URLS.general} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 underline">Learn more <ExternalLink className="h-3 w-3" /></a>
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Agent Name */}
                <div className="space-y-2">
                    <Label htmlFor="workflow_name" className="text-sm font-medium">Agent Name</Label>
                    <Input
                        id="workflow_name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Enter Agent name"
                    />
                </div>

                <Separator />

                {/* Ambient Noise */}
                <div className="space-y-4">
                    <div>
                        <h3 className="text-sm font-medium">Ambient Noise</h3>
                        <p className="text-xs text-muted-foreground mt-0.5">
                            Add background ambient noise to make the conversation sound more natural.
                        </p>
                    </div>
                    <div className="flex items-center justify-between">
                        <Label htmlFor="ambient-noise-enabled" className="text-sm">Use Ambient Noise</Label>
                        <Switch
                            id="ambient-noise-enabled"
                            checked={ambientNoiseConfig.enabled}
                            onCheckedChange={(checked) => {
                                unrevert(GENERAL_LEAVES.ambientEnabled);
                                setAmbientNoiseConfig((prev) => ({ ...prev, enabled: checked }));
                            }}
                        />
                    </div>
                    {ambientNoiseConfig.enabled && (
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="ambient-volume" className="text-xs">Volume</Label>
                                <Input
                                    id="ambient-volume"
                                    type="number"
                                    step="0.1"
                                    min="0"
                                    max="1"
                                    value={ambientNoiseConfig.volume}
                                    onChange={(e) => {
                                        const value = parseFloat(e.target.value);
                                        if (isNaN(value)) return;
                                        unrevert(GENERAL_LEAVES.ambientVolume);
                                        setAmbientNoiseConfig((prev) => ({ ...prev, volume: value }));
                                    }}
                                />
                            </div>

                            {/* Custom Audio File */}
                            <div className="space-y-2">
                                <Label className="text-xs">Custom Audio File</Label>
                                <p className="text-xs text-muted-foreground">
                                    Upload your own audio file or use the default office ambience.
                                </p>

                                {ambientNoiseConfig.storage_key ? (
                                    <div className="flex items-center gap-2 rounded-md border p-2 bg-muted/10">
                                        <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono truncate flex-1">
                                            {ambientNoiseConfig.original_filename || "Custom audio"}
                                        </code>
                                        <Button
                                            type="button"
                                            size="sm"
                                            variant="ghost"
                                            className="h-6 w-6 p-0 shrink-0"
                                            onClick={async () => {
                                                try {
                                                    await togglePlayback(
                                                        "ambient-noise",
                                                        ambientNoiseConfig.storage_key!,
                                                        ambientNoiseConfig.storage_backend,
                                                    );
                                                } catch {
                                                    setAudioUploadError("Failed to play audio");
                                                }
                                            }}
                                        >
                                            {playingId === "ambient-noise" ? (
                                                <Pause className="w-3.5 h-3.5" />
                                            ) : (
                                                <Play className="w-3.5 h-3.5" />
                                            )}
                                        </Button>
                                        <Button
                                            type="button"
                                            size="sm"
                                            variant="ghost"
                                            className="h-6 w-6 p-0 shrink-0"
                                            onClick={handleRemoveCustomAudio}
                                        >
                                            <X className="w-3.5 h-3.5" />
                                        </Button>
                                    </div>
                                ) : (
                                    <div>
                                        <input
                                            ref={ambientFileInputRef}
                                            type="file"
                                            accept="audio/*"
                                            onChange={(e) => {
                                                const file = e.target.files?.[0];
                                                if (file) handleAmbientFileUpload(file);
                                            }}
                                            className="hidden"
                                        />
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            className="text-sm font-normal"
                                            onClick={() => ambientFileInputRef.current?.click()}
                                            disabled={isUploadingAudio}
                                        >
                                            {isUploadingAudio ? (
                                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                            ) : (
                                                <Upload className="w-4 h-4 mr-2" />
                                            )}
                                            {isUploadingAudio ? "Uploading..." : "Upload audio file (max 10MB)"}
                                        </Button>
                                    </div>
                                )}

                                {audioUploadError && (
                                    <p className="text-xs text-destructive">{audioUploadError}</p>
                                )}

                                {!ambientNoiseConfig.storage_key && (
                                    <p className="text-xs text-muted-foreground italic">
                                        Using default office ambience
                                    </p>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <Separator />

                {/* Turn Detection */}
                <div className="space-y-4">
                    <div>
                        <h3 className="text-sm font-medium">Turn Detection</h3>
                        <p className="text-xs text-muted-foreground mt-0.5">
                            Configure how the agent detects when the user has finished speaking.
                        </p>
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="turn_stop_strategy" className="text-xs">Detection Strategy</Label>
                        <Select
                            value={turnStopStrategy}
                            onValueChange={(value: TurnStopStrategy) => {
                                unrevert(GENERAL_LEAVES.turnStopStrategy);
                                setTurnStopStrategy(value);
                            }}
                        >
                            <SelectTrigger id="turn_stop_strategy">
                                <SelectValue placeholder="Select strategy" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="transcription">Transcription-based</SelectItem>
                                <SelectItem value="turn_analyzer">Smart Turn Analyzer</SelectItem>
                            </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground">
                            {turnStopStrategy === "transcription"
                                ? "Best for short responses (1-2 word statements). Ends turn when transcription indicates completion."
                                : "Best for longer responses with natural pauses. Uses ML model to detect end of turn."}
                        </p>
                    </div>
                    {turnStopStrategy === "turn_analyzer" && (
                        <div className="space-y-2">
                            <Label htmlFor="smart_turn_stop_secs" className="text-xs">
                                Incomplete Turn Timeout (seconds)
                            </Label>
                            <Input
                                id="smart_turn_stop_secs"
                                type="number"
                                step="0.5"
                                min="0.5"
                                max="10"
                                value={smartTurnStopSecs}
                                onChange={(e) => {
                                    const value = parseFloat(e.target.value);
                                    if (isNaN(value) || value < 0.5) return;
                                    unrevert(GENERAL_LEAVES.smartTurnStopSecs);
                                    setSmartTurnStopSecs(value);
                                }}
                            />
                            <p className="text-xs text-muted-foreground">
                                Max silence duration before ending an incomplete turn. Default: 2 seconds
                            </p>
                        </div>
                    )}
                </div>

                <Separator />

                {/* Interruption */}
                <div className="space-y-4">
                    <div>
                        <h3 className="text-sm font-medium">Interruption</h3>
                        <p className="text-xs text-muted-foreground mt-0.5">
                            Configure when user speech should interrupt the agent while it is speaking.
                        </p>
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="turn_start_strategy" className="text-xs">Interruption Strategy</Label>
                        <Select
                            value={turnStartStrategy}
                            onValueChange={(value: TurnStartStrategy) => {
                                unrevert(GENERAL_LEAVES.turnStartStrategy);
                                setTurnStartStrategy(value);
                            }}
                        >
                            <SelectTrigger id="turn_start_strategy">
                                <SelectValue placeholder="Select strategy" />
                            </SelectTrigger>
                            <SelectContent>
                                {TURN_START_STRATEGY_OPTIONS.map((option) => (
                                    <SelectItem key={option.value} value={option.value}>
                                        {option.label}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground">
                            {selectedTurnStartStrategy?.description}
                            {turnStartStrategy === "provisional_vad" && (
                                <span className="ml-2 inline-flex rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                                    Experimental
                                </span>
                            )}
                        </p>
                    </div>
                    {turnStartStrategy === "min_words" && (
                        <div className="space-y-2">
                            <Label htmlFor="turn_start_min_words" className="text-xs">
                                Minimum Words Before Interruption
                            </Label>
                            <Input
                                id="turn_start_min_words"
                                type="number"
                                step="1"
                                min="1"
                                max="10"
                                value={turnStartMinWords}
                                onChange={(e) => {
                                    const value = parseInt(e.target.value);
                                    if (isNaN(value) || value < 1) return;
                                    unrevert(GENERAL_LEAVES.turnStartMinWords);
                                    setTurnStartMinWords(value);
                                }}
                            />
                            <p className="text-xs text-muted-foreground">
                                Number of transcribed words needed to interrupt while the bot is speaking. Default: {DEFAULT_TURN_START_MIN_WORDS}
                            </p>
                        </div>
                    )}
                    {turnStartStrategy === "provisional_vad" && (
                        <div className="space-y-2">
                            <Label htmlFor="provisional_vad_pause_secs" className="text-xs">
                                Provisional Pause (seconds)
                            </Label>
                            <Input
                                id="provisional_vad_pause_secs"
                                type="number"
                                step="0.1"
                                min="0.1"
                                max="5"
                                value={provisionalVadPauseSecs}
                                onChange={(e) => {
                                    const value = parseFloat(e.target.value);
                                    if (isNaN(value) || value < 0.1) return;
                                    unrevert(GENERAL_LEAVES.provisionalVadPauseSecs);
                                    setProvisionalVadPauseSecs(value);
                                }}
                            />
                            <p className="text-xs text-muted-foreground">
                                Seconds to pause bot audio while waiting for transcript confirmation. Default: {DEFAULT_PROVISIONAL_VAD_PAUSE_SECS}
                            </p>
                        </div>
                    )}
                </div>

                <Separator />

                {/* Transcript */}
                <div className="space-y-4">
                    <div>
                        <h3 className="text-sm font-medium">Transcript</h3>
                        <p className="text-xs text-muted-foreground mt-0.5">
                            Include start and stop timestamps for each speaker in the uploaded transcript.
                        </p>
                    </div>
                    <div className="flex items-center justify-between">
                        <Label htmlFor="transcript-end-timestamps-enabled" className="text-sm">
                            Enhanced Timestamped Transcript
                        </Label>
                        <Switch
                            id="transcript-end-timestamps-enabled"
                            checked={includeTranscriptEndTimestamps}
                            onCheckedChange={(checked) => {
                                unrevert(GENERAL_LEAVES.transcriptEndTimestamps);
                                setIncludeTranscriptEndTimestamps(checked);
                            }}
                        />
                    </div>
                    <div className="rounded-md border bg-muted/20 p-3">
                        <pre className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
                            {`[2026-07-06T10:00:00.000Z -> 2026-07-06T10:00:04.800Z] assistant: Can you confirm your date of birth?
[2026-07-06T10:00:06.200Z -> 2026-07-06T10:00:08.700Z] user: January fifth, nineteen ninety.`}
                        </pre>
                    </div>
                </div>

                <Separator />

                {/* Context Compaction */}
                <div className="space-y-4">
                    <div>
                        <h3 className="text-sm font-medium">Context Compaction</h3>
                        <p className="text-xs text-muted-foreground mt-0.5">
                            Automatically summarize conversation context when transitioning between nodes. Not applicable in Realtime mode - the speech-to-speech service manages its own conversation state and this setting is ignored.
                        </p>
                    </div>
                    <div className="flex items-center justify-between">
                        <Label htmlFor="context-compaction-enabled" className="text-sm">
                            Enable Context Compaction
                        </Label>
                        <Switch
                            id="context-compaction-enabled"
                            checked={contextCompactionEnabled}
                            onCheckedChange={(checked) => {
                                unrevert(GENERAL_LEAVES.contextCompactionEnabled);
                                setContextCompactionEnabled(checked);
                            }}
                        />
                    </div>
                </div>

                <Separator />

                <CallDispositionEditor
                    rows={callDispositionRows}
                    onChange={(rows) => {
                        unrevert(GENERAL_LEAVES.callDispositions);
                        setCallDispositionRows(rows);
                    }}
                    defaultDispositions={defaultCallDispositions}
                />

                <Separator />

                {/* Call Management */}
                <div className="space-y-4">
                    <div>
                        <h3 className="text-sm font-medium">Call Management</h3>
                        <p className="text-xs text-muted-foreground mt-0.5">
                            Configure call duration limits and idle timeout settings.
                        </p>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="max_call_duration" className="text-xs">Max Call Duration (seconds)</Label>
                            <Input
                                id="max_call_duration"
                                type="number"
                                min="1"
                                value={maxCallDuration}
                                onChange={(e) => {
                                    const value = parseInt(e.target.value);
                                    if (isNaN(value) || value <= 0) return;
                                    unrevert(GENERAL_LEAVES.maxCallDuration);
                                    setMaxCallDuration(value);
                                }}
                            />
                            <p className="text-xs text-muted-foreground">Default: 600 (10 minutes)</p>
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="max_user_idle_timeout" className="text-xs">
                                Max User Idle Timeout (seconds)
                            </Label>
                            <Input
                                id="max_user_idle_timeout"
                                type="number"
                                min="1"
                                value={maxUserIdleTimeout}
                                onChange={(e) => {
                                    const value = parseInt(e.target.value);
                                    if (isNaN(value) || value <= 0) return;
                                    unrevert(GENERAL_LEAVES.maxUserIdleTimeout);
                                    setMaxUserIdleTimeout(value);
                                }}
                            />
                            <p className="text-xs text-muted-foreground">Default: 10 seconds</p>
                        </div>
                    </div>
                </div>

                {externalPbxIntegrationsEnabled && (
                    <>
                        <Separator />

                        {/* External PBX Field Updates */}
                        <div className="space-y-4">
                            <div>
                                <h3 className="text-sm font-medium">External PBX Field Updates</h3>
                                <p className="text-xs text-muted-foreground mt-0.5">
                                    Optionally copy final gathered-context values into provider-native fields before transfer or hangup.
                                </p>
                            </div>
                            <div className="flex items-center justify-between">
                                <Label className="text-sm">Field Mappings</Label>
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => setExternalPbxFieldMappings((current) => [
                                        ...current,
                                        { context_path: "", destination_field: "" },
                                    ])}
                                >
                                    <Plus className="mr-1 h-4 w-4" /> Add mapping
                                </Button>
                            </div>
                            <div className="space-y-2">
                                {externalPbxFieldMappings.map((mapping, index) => (
                                    <div key={index} className="grid grid-cols-[1fr_1fr_auto] gap-2">
                                        <Input
                                            aria-label={`Gathered context field ${index + 1}`}
                                            value={mapping.context_path}
                                            onChange={(event) => setExternalPbxFieldMappings((current) =>
                                                current.map((item, itemIndex) =>
                                                    itemIndex === index
                                                        ? { ...item, context_path: event.target.value }
                                                        : item,
                                                )
                                            )}
                                            placeholder="qualified"
                                        />
                                        <Input
                                            aria-label={`External PBX destination field ${index + 1}`}
                                            value={mapping.destination_field}
                                            onChange={(event) => setExternalPbxFieldMappings((current) =>
                                                current.map((item, itemIndex) =>
                                                    itemIndex === index
                                                        ? { ...item, destination_field: event.target.value }
                                                        : item,
                                                )
                                            )}
                                            placeholder="address3"
                                        />
                                        <Button
                                            type="button"
                                            variant="ghost"
                                            size="icon"
                                            aria-label={`Remove external PBX field mapping ${index + 1}`}
                                            onClick={() => setExternalPbxFieldMappings((current) =>
                                                current.filter((_, itemIndex) => itemIndex !== index)
                                            )}
                                        >
                                            <Trash2Icon className="h-4 w-4" />
                                        </Button>
                                    </div>
                                ))}
                                {externalPbxFieldMappings.length === 0 && (
                                    <p className="text-xs text-muted-foreground">
                                        No external fields will be updated. Context names may be direct extracted-variable names or paths such as extracted_variables.qualified.
                                    </p>
                                )}
                                {!externalPbxFieldMappingsValid && (
                                    <p className="text-xs text-destructive">
                                        Each mapping needs a context field and a destination field containing only letters, numbers, and underscores.
                                    </p>
                                )}
                            </div>

                            <div className="space-y-4 border-t pt-4">
                                <div>
                                    <h3 className="text-sm font-medium">Lead Fields To Capture</h3>
                                    <p className="text-xs text-muted-foreground mt-0.5">
                                        Extra lead fields to read from the inbound call, named without the header prefix
                                        (<code>first_name</code> reads <code>X-VICIDIAL-first_name</code>). Captured values are
                                        addressable in prompts as <code>{"{{initial_context.external_pbx_call.lead.<field>}}"}</code>.
                                        Each field adds one request during call setup, so list only what the agent uses.
                                    </p>
                                </div>
                                <div className="flex items-center justify-between">
                                    <Label className="text-sm">Lead Fields</Label>
                                    <Button
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setExternalPbxLeadHeaders((current) => [...current, ""])}
                                    >
                                        <Plus className="mr-1 h-4 w-4" /> Add field
                                    </Button>
                                </div>
                                <div className="space-y-2">
                                    {externalPbxLeadHeaders.map((field, index) => (
                                        <div key={index} className="grid grid-cols-[1fr_auto] gap-2">
                                            <Input
                                                aria-label={`External PBX lead field ${index + 1}`}
                                                value={field}
                                                onChange={(event) => setExternalPbxLeadHeaders((current) =>
                                                    current.map((item, itemIndex) =>
                                                        itemIndex === index ? event.target.value : item,
                                                    )
                                                )}
                                                placeholder="first_name"
                                            />
                                            <Button
                                                type="button"
                                                variant="ghost"
                                                size="icon"
                                                aria-label={`Remove external PBX lead field ${index + 1}`}
                                                onClick={() => setExternalPbxLeadHeaders((current) =>
                                                    current.filter((_, itemIndex) => itemIndex !== index)
                                                )}
                                            >
                                                <Trash2Icon className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    ))}
                                    {externalPbxLeadHeaders.length === 0 && (
                                        <p className="text-xs text-muted-foreground">
                                            Only the identity fields needed to transfer or hang up the call are captured.
                                        </p>
                                    )}
                                    {!externalPbxLeadHeadersValid && (
                                        <p className="text-xs text-destructive">
                                            Each lead field must start with a letter and contain only letters, numbers, and underscores.
                                        </p>
                                    )}
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </CardContent>
            <CardFooter className="justify-end gap-3 border-t pt-6">
                {isDirty && <span className="text-xs text-muted-foreground">Unsaved changes</span>}
                <Button
                    onClick={handleSave}
                    disabled={
                        isSaving
                        || !isDirty
                        || !callDispositionsValid
                        || (externalPbxIntegrationsEnabled && !externalPbxSettingsValid)
                    }
                >
                    {isSaving ? "Saving..." : "Save General Settings"}
                </Button>
            </CardFooter>
        </Card>
    );
}

// ---------------------------------------------------------------------------
// Section: Template Variables
// ---------------------------------------------------------------------------

function TemplateVariablesSection({
    templateContextVariables,
    onSave,
}: {
    templateContextVariables: Record<string, string>;
    onSave: (variables: Record<string, string>) => Promise<void>;
}) {
    const [contextVars, setContextVars] = useState<Record<string, string>>(templateContextVariables);
    const [newKey, setNewKey] = useState("");
    const [newValue, setNewValue] = useState("");
    const [isSaving, setIsSaving] = useState(false);

    const isDirty = useMemo(() => {
        const pendingVars = newKey && newValue ? { ...contextVars, [newKey]: newValue } : contextVars;
        return JSON.stringify(pendingVars) !== JSON.stringify(templateContextVariables);
    }, [contextVars, newKey, newValue, templateContextVariables]);

    useUnsavedChanges("variables", isDirty);

    const handleAdd = () => {
        if (newKey && newValue) {
            setContextVars((prev) => ({ ...prev, [newKey]: newValue }));
        }
        setNewKey("");
        setNewValue("");
    };

    const handleRemove = (key: string) => {
        setContextVars((prev) => {
            const next = { ...prev };
            delete next[key];
            return next;
        });
    };

    const handleSave = async () => {
        setIsSaving(true);
        try {
            let varsToSave = contextVars;
            if (newKey && newValue) {
                varsToSave = { ...varsToSave, [newKey]: newValue };
            }
            await onSave(varsToSave);
            toast.success(`Template variables saved. ${PUBLISH_WORKFLOW_REMINDER}`);
        } catch (error) {
            console.error("Failed to save variables:", error);
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <Card id="variables">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                    <Variable className="h-4 w-4" />
                    Template Variables
                </CardTitle>
                <CardDescription>
                    Variables available in workflow prompts via {`{{variable_name}}`} syntax for testing the workflow.{" "}
                    <a href={SETTINGS_DOCUMENTATION_URLS.templateVariables} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 underline">Learn more <ExternalLink className="h-3 w-3" /></a>
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {/* Existing Variables */}
                {Object.entries(contextVars).length > 0 && (
                    <div className="space-y-2">
                        <Label className="text-sm font-medium">Current Variables</Label>
                        {Object.entries(contextVars).map(([key, value]) => (
                            <div key={key} className="flex items-center gap-2 rounded-md border p-2">
                                <div className="flex-1 min-w-0">
                                    <div className="text-sm font-medium">{key}</div>
                                    <div className="text-xs text-muted-foreground truncate">{value}</div>
                                </div>
                                <Button size="sm" variant="ghost" onClick={() => handleRemove(key)}>
                                    <Trash2Icon className="h-4 w-4" />
                                </Button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Add New Variable */}
                <div className="space-y-3">
                    <Label className="text-sm font-medium">Add New Variable</Label>
                    <div className="flex gap-2">
                        <div className="flex-1 space-y-1">
                            <Label htmlFor="var-key" className="text-xs">Key</Label>
                            <Input
                                id="var-key"
                                placeholder="Enter variable key"
                                value={newKey}
                                onChange={(e) => setNewKey(e.target.value)}
                            />
                        </div>
                        <div className="flex-1 space-y-1">
                            <Label htmlFor="var-value" className="text-xs">Value</Label>
                            <Input
                                id="var-value"
                                placeholder="Enter variable value"
                                value={newValue}
                                onChange={(e) => setNewValue(e.target.value)}
                            />
                        </div>
                    </div>
                    <Button size="sm" onClick={handleAdd} disabled={!newKey || !newValue}>
                        Add Variable
                    </Button>
                </div>
            </CardContent>
            <CardFooter className="justify-end gap-3 border-t pt-6">
                {isDirty && <span className="text-xs text-muted-foreground">Unsaved changes</span>}
                <Button onClick={handleSave} disabled={isSaving || !isDirty}>
                    {isSaving ? "Saving..." : "Save Variables"}
                </Button>
            </CardFooter>
        </Card>
    );
}

// ---------------------------------------------------------------------------
// Section: Dictionary
// ---------------------------------------------------------------------------

function DictionarySection({
    configuration,
    onSave,
}: {
    configuration: WorkflowConfigurationState;
    onSave: (patch: ConfigurationPatch, workflowName?: string) => Promise<void>;
}) {
    const stored = configuration.effective.dictionary ?? "";
    const [dictionaryValue, setDictionaryValue] = useState(stored);
    const [pendingRevert, setPendingRevert] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    const isDirty = pendingRevert || dictionaryValue !== stored;

    useUnsavedChanges("dictionary", isDirty);

    const handleSave = async () => {
        setIsSaving(true);
        try {
            await onSave(
                pendingRevert
                    ? { set: [], unset: [["dictionary"]] }
                    : { set: [{ path: ["dictionary"], value: dictionaryValue }], unset: [] },
            );
            toast.success(`Dictionary saved. ${PUBLISH_WORKFLOW_REMINDER}`);
        } catch (error) {
            console.error("Failed to save dictionary:", error);
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <Card id="dictionary">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                    <BookA className="h-4 w-4" />
                    Dictionary
                </CardTitle>
                <CardDescription>
                    Add words the agent should actively listen for &mdash; company jargon, names,
                    industry terms. May incur extra cost depending on provider.
                </CardDescription>
            </CardHeader>
            <CardContent>
                <Textarea
                    placeholder="Enter words separated by comma (e.g. billing department, tretinoin)"
                    value={dictionaryValue}
                    onChange={(e) => {
                        setPendingRevert(false);
                        setDictionaryValue(e.target.value);
                    }}
                    rows={4}
                    className="resize-none"
                />
            </CardContent>
            <CardFooter className="justify-end gap-3 border-t pt-6">
                {isDirty && <span className="text-xs text-muted-foreground">Unsaved changes</span>}
                <Button onClick={handleSave} disabled={isSaving || !isDirty}>
                    {isSaving ? "Saving..." : "Save Dictionary"}
                </Button>
            </CardFooter>
        </Card>
    );
}

// ---------------------------------------------------------------------------
// Section: Voicemail Detection
// ---------------------------------------------------------------------------

function VoicemailSection({
    configuration,
    onSave,
}: {
    configuration: WorkflowConfigurationState;
    onSave: (patch: ConfigurationPatch, workflowName?: string) => Promise<void>;
}) {
    const voicemailDetection = configuration.effective.voicemail_detection;
    const getConfig = (): VoicemailDetectionConfiguration => ({
        ...DEFAULT_VOICEMAIL_DETECTION_CONFIGURATION,
        ...voicemailDetection,
    });

    const [enabled, setEnabled] = useState(getConfig().enabled);
    const [useWorkflowLlm, setUseWorkflowLlm] = useState(getConfig().use_workflow_llm);
    const [provider, setProvider] = useState(getConfig().provider || "openai");
    const [model, setModel] = useState(getConfig().model || "gpt-4.1");
    const [apiKey, setApiKey] = useState(getConfig().api_key || "");
    const [systemPrompt, setSystemPrompt] = useState(getConfig().system_prompt || DEFAULT_VOICEMAIL_SYSTEM_PROMPT);
    const [longSpeechTimeout, setLongSpeechTimeout] = useState(getConfig().long_speech_timeout);
    // Set ⇒ the save drops the workflow's own voicemail block and inherits again.
    const [pendingRevert, setPendingRevert] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    const isDirty = useMemo(() => {
        const init = {
            ...DEFAULT_VOICEMAIL_DETECTION_CONFIGURATION,
            ...voicemailDetection,
        };
        return (
            pendingRevert ||
            enabled !== init.enabled ||
            useWorkflowLlm !== init.use_workflow_llm ||
            provider !== (init.provider || "openai") ||
            model !== (init.model || "gpt-4.1") ||
            apiKey !== (init.api_key || "") ||
            systemPrompt !== (init.system_prompt || DEFAULT_VOICEMAIL_SYSTEM_PROMPT) ||
            longSpeechTimeout !== init.long_speech_timeout
        );
    }, [pendingRevert, enabled, useWorkflowLlm, provider, model, apiKey, systemPrompt, longSpeechTimeout, voicemailDetection]);

    useUnsavedChanges("voicemail", isDirty);

    // The whole block is one leaf, so any edit cancels a pending revert.
    const edit = <T,>(setter: (value: T) => void) => (value: T) => {
        setPendingRevert(false);
        setter(value);
    };

    const handleSave = async () => {
        setIsSaving(true);
        try {
            const voicemailConfig: VoicemailDetectionConfiguration = {
                enabled,
                use_workflow_llm: useWorkflowLlm,
                provider: useWorkflowLlm ? undefined : provider,
                model: useWorkflowLlm ? undefined : model,
                api_key: useWorkflowLlm ? undefined : apiKey,
                system_prompt:
                    systemPrompt && systemPrompt !== DEFAULT_VOICEMAIL_SYSTEM_PROMPT ? systemPrompt : undefined,
                long_speech_timeout: longSpeechTimeout,
            };
            await onSave(
                pendingRevert
                    ? { set: [], unset: [["voicemail_detection"]] }
                    : { set: [{ path: ["voicemail_detection"], value: voicemailConfig }], unset: [] },
            );
            toast.success(`Voicemail settings saved. ${PUBLISH_WORKFLOW_REMINDER}`);
        } catch (error) {
            console.error("Failed to save voicemail settings:", error);
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <Card id="voicemail">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                    <PhoneOff className="h-4 w-4" />
                    Voicemail Detection
                </CardTitle>
                <CardDescription>
                    Automatically detect and end calls when a voicemail system is reached.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex items-center space-x-2 rounded-md border bg-muted/20 p-2">
                    <Switch id="voicemail-enabled" checked={enabled} onCheckedChange={edit(setEnabled)} />
                    <Label htmlFor="voicemail-enabled">Enable Voicemail Detection</Label>
                </div>

                {enabled && (
                    <>
                        {/* LLM Configuration */}
                        <div className="space-y-3">
                            <div className="flex items-center space-x-2 rounded-md border bg-muted/20 p-2">
                                <Switch
                                    id="voicemail-use-workflow-llm"
                                    checked={useWorkflowLlm}
                                    onCheckedChange={edit(setUseWorkflowLlm)}
                                />
                                <Label htmlFor="voicemail-use-workflow-llm">Use Workflow LLM</Label>
                                <Label className="ml-2 text-xs text-muted-foreground">
                                    Use the LLM configured in your account settings.
                                </Label>
                            </div>

                            {!useWorkflowLlm && (
                                <LLMConfigSelector
                                    provider={provider}
                                    onProviderChange={edit(setProvider)}
                                    model={model}
                                    onModelChange={edit(setModel)}
                                    apiKey={apiKey}
                                    onApiKeyChange={edit(setApiKey)}
                                />
                            )}
                        </div>

                        {/* System Prompt */}
                        <div className="space-y-2">
                            <Label>System Prompt</Label>
                            <p className="text-xs text-muted-foreground">
                                The LLM must respond with either &quot;CONVERSATION&quot; or &quot;VOICEMAIL&quot;.
                            </p>
                            <Textarea
                                value={systemPrompt}
                                onChange={(e) => edit(setSystemPrompt)(e.target.value)}
                                className="min-h-[200px] font-mono text-xs"
                            />
                        </div>

                        {/* Timing */}
                        <div className="space-y-2 rounded-md border bg-muted/10 p-3">
                            <Label className="font-medium">Timing</Label>
                            <div className="space-y-2">
                                <Label className="text-sm">Speech Cutoff (seconds)</Label>
                                <p className="text-xs text-muted-foreground">
                                    Trigger classification early if first turn speech exceeds this duration.
                                </p>
                                <Input
                                    type="number"
                                    step="0.5"
                                    min="1"
                                    max="30"
                                    value={longSpeechTimeout}
                                    onChange={(e) => edit(setLongSpeechTimeout)(parseFloat(e.target.value) || 8.0)}
                                />
                            </div>
                        </div>
                    </>
                )}
            </CardContent>
            <CardFooter className="justify-end gap-3 border-t pt-6">
                {isDirty && <span className="text-xs text-muted-foreground">Unsaved changes</span>}
                <Button onClick={handleSave} disabled={isSaving || !isDirty}>
                    {isSaving ? "Saving..." : "Save Voicemail Settings"}
                </Button>
            </CardFooter>
        </Card>
    );
}

// ---------------------------------------------------------------------------
// Section: Agent UUID
// ---------------------------------------------------------------------------

function AgentUuidSection({ workflowUuid }: { workflowUuid: string }) {
    const handleCopy = async () => {
        try {
            await copyTextToClipboard(workflowUuid);
            toast.success("Agent UUID copied");
        } catch {
            toast.error("Failed to copy Agent UUID");
        }
    };

    return (
        <Card id="identity">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                    <Fingerprint className="h-4 w-4" />
                    Agent UUID
                </CardTitle>
                <CardDescription>
                    Stable identifier for this agent. Used in agent-stream URLs and
                    other integrations where a numeric workflow ID isn&apos;t portable.
                </CardDescription>
            </CardHeader>
            <CardContent>
                <button
                    type="button"
                    onClick={handleCopy}
                    title="Click to copy"
                    className="group flex w-full items-center gap-2 rounded-md border bg-muted/20 p-2 text-left font-mono text-xs transition-colors hover:bg-muted/40"
                >
                    <code className="flex-1 truncate">{workflowUuid}</code>
                    <Clipboard className="h-3.5 w-3.5 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground" />
                </button>
            </CardContent>
            <CardFooter className="border-t pt-6">
                <Button variant="outline" size="sm" onClick={handleCopy}>
                    <Clipboard className="h-3.5 w-3.5 mr-2" />
                    Copy UUID
                </Button>
            </CardFooter>
        </Card>
    );
}

// ---------------------------------------------------------------------------
// Section: Model Overrides
// ---------------------------------------------------------------------------

function WorkflowModelOverridesSection({
    configuration,
    onSave,
    modelConfigurationDefaults,
    organizationModelConfiguration,
    modelConfigurationPricing,
    modelConfigurationLoading,
    modelConfigurationError,
}: {
    configuration: WorkflowConfigurationState;
    onSave: (patch: ConfigurationPatch, workflowName?: string) => Promise<void>;
    modelConfigurationDefaults: ModelConfigurationDefaultsV2 | null;
    organizationModelConfiguration: OrganizationAiModelConfigurationResponse | null;
    modelConfigurationPricing: ModelConfigurationPricingResponse | null;
    modelConfigurationLoading: boolean;
    modelConfigurationError: string | null;
}) {
    // Read from `own`: model overrides never cascade from the organization, so
    // `own` is the honest answer to "does this workflow override the models?".
    const savedV2Override = configuration.own.model_configuration_v2_override as
        OrganizationAiModelConfigurationV2 | undefined;
    const hasSavedModelOverride = Boolean(savedV2Override || configuration.own.model_overrides);
    const [overrideEnabled, setOverrideEnabled] = useState(Boolean(savedV2Override));
    const [isRemovingOverride, setIsRemovingOverride] = useState(false);

    useEffect(() => {
        setOverrideEnabled(Boolean(configuration.own.model_configuration_v2_override));
    }, [configuration.own.model_configuration_v2_override]);

    const hasOrgConfiguration = organizationModelConfiguration?.source === "organization_v2";

    const saveV2Override = async (modelConfiguration: OrganizationAiModelConfigurationV2) => {
        await onSave({
            set: [{ path: ["model_configuration_v2_override"], value: modelConfiguration }],
            unset: [["model_overrides"]],
        });
        toast.success(`Model override saved. ${PUBLISH_WORKFLOW_REMINDER}`);
    };

    const removeV2Override = async () => {
        setIsRemovingOverride(true);
        try {
            await onSave({
                set: [],
                unset: [["model_configuration_v2_override"], ["model_overrides"]],
            });
            setOverrideEnabled(false);
            toast.success(`Organization model configuration saved. ${PUBLISH_WORKFLOW_REMINDER}`);
        } finally {
            setIsRemovingOverride(false);
        }
    };

    return (
        <Card id="models">
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                    <Brain className="h-4 w-4" />
                    Model Overrides
                </CardTitle>
                <CardDescription>
                    Override the full organization model configuration for this workflow.{" "}
                    <a href={SETTINGS_DOCUMENTATION_URLS.modelOverrides} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 underline">Learn more <ExternalLink className="h-3 w-3" /></a>
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {modelConfigurationLoading && (
                    <div className="flex items-center gap-2 rounded-md border p-4 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading model configuration
                    </div>
                )}

                {modelConfigurationError && (
                    <div className="rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                        {modelConfigurationError}
                    </div>
                )}

                {!modelConfigurationLoading && !modelConfigurationError && !hasOrgConfiguration && (
                    <div className="flex flex-col gap-3 rounded-md border bg-muted/30 p-4 sm:flex-row sm:items-center sm:justify-between">
                        <p className="text-sm text-muted-foreground">
                            Set up your organization model configuration before overriding it per workflow.
                        </p>
                        <Button type="button" variant="outline" size="sm" asChild>
                            <Link href="/model-configurations">Configure Models</Link>
                        </Button>
                    </div>
                )}

                {!modelConfigurationLoading && !modelConfigurationError && hasOrgConfiguration && modelConfigurationDefaults && organizationModelConfiguration && (
                    <>
                        <div className="flex items-center justify-between rounded-md border p-4">
                            <div className="space-y-0.5">
                                <Label htmlFor="workflow-model-v2-override" className="text-sm font-medium">
                                    Override for this workflow
                                </Label>
                                <p className="text-xs text-muted-foreground">
                                    {overrideEnabled
                                        ? "This workflow uses its own complete model configuration."
                                        : "This workflow uses the organization model configuration."}
                                </p>
                            </div>
                            <Switch
                                id="workflow-model-v2-override"
                                checked={overrideEnabled}
                                onCheckedChange={setOverrideEnabled}
                            />
                        </div>

                        {overrideEnabled ? (
                            <AIModelConfigurationV2Editor
                                defaults={modelConfigurationDefaults}
                                configuration={
                                    (savedV2Override as OrganizationAiModelConfigurationV2 | undefined)
                                    || (organizationModelConfiguration.configuration as OrganizationAiModelConfigurationV2 | null)
                                }
                                effectiveConfiguration={
                                    savedV2Override
                                        ? null
                                        : organizationModelConfiguration.effective_configuration
                                }
                                pricing={modelConfigurationPricing}
                                submitLabel="Save Model Override"
                                onSave={saveV2Override}
                            />
                        ) : (
                            <div className="rounded-md border bg-muted/20 p-4">
                                <p className="text-sm text-muted-foreground">
                                    Using organization model configuration.
                                </p>
                                {hasSavedModelOverride && (
                                    <Button
                                        type="button"
                                        className="mt-3"
                                        onClick={removeV2Override}
                                        disabled={isRemovingOverride}
                                    >
                                        {isRemovingOverride ? "Saving..." : "Save Organization Configuration"}
                                    </Button>
                                )}
                            </div>
                        )}
                    </>
                )}
            </CardContent>
        </Card>
    );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Page wrapper — handles auth & data fetching, then mounts the content
// component only when everything is loaded. This avoids useWorkflowState
// running with empty initial values and overwriting the Zustand store.
// ---------------------------------------------------------------------------

export default function WorkflowSettingsPage() {
    const params = useParams();
    const { user, redirectToLogin, loading: authLoading } = useAuth();
    const [workflow, setWorkflow] = useState<WorkflowResponse | undefined>(undefined);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!authLoading && !user) {
            redirectToLogin();
        }
    }, [authLoading, user, redirectToLogin]);

    useEffect(() => {
        const fetchWorkflow = async () => {
            if (!user) return;
            try {
                const response = await getWorkflowApiV1WorkflowFetchWorkflowIdGet({
                    path: { workflow_id: Number(params.workflowId) },
                });
                setWorkflow(response.data);
            } catch (err) {
                setError("Failed to fetch workflow");
                logger.error(`Error fetching workflow settings: ${err}`);
            } finally {
                setLoading(false);
            }
        };
        if (user) fetchWorkflow();
    }, [params.workflowId, user]);

    if (loading || authLoading) return <SpinLoader />;

    if (error || !workflow) {
        return (
            <div className="flex min-h-screen items-center justify-center">
                <div className="text-lg text-destructive">{error || "Workflow not found"}</div>
            </div>
        );
    }

    if (!user) return null;

    return <WorkflowSettingsContent workflow={workflow} user={user} />;
}

// ---------------------------------------------------------------------------
// Content — only mounts once the workflow API response is available, so
// useWorkflowState always initialises with real data.
// ---------------------------------------------------------------------------

function WorkflowSettingsContent({
    workflow,
    user,
}: {
    workflow: WorkflowResponse;
    user: { id: string; email?: string };
}) {
    return (
        <UnsavedChangesProvider>
            <WorkflowSettingsInner workflow={workflow} user={user} />
        </UnsavedChangesProvider>
    );
}

function WorkflowSettingsInner({
    workflow,
    user,
}: {
    workflow: WorkflowResponse;
    user: { id: string; email?: string };
}) {
    const router = useRouter();
    const { dirtySections, confirmNavigate } = useUnsavedChangesContext();

    const [isEmbedDialogOpen, setIsEmbedDialogOpen] = useState(false);
    const [activeSection, setActiveSection] = useState("general");
    const [modelConfigurationDefaults, setModelConfigurationDefaults] = useState<ModelConfigurationDefaultsV2 | null>(null);
    const [organizationModelConfiguration, setOrganizationModelConfiguration] = useState<OrganizationAiModelConfigurationResponse | null>(null);
    const [modelConfigurationPricing, setModelConfigurationPricing] = useState<ModelConfigurationPricingResponse | null>(null);
    const [modelConfigurationLoading, setModelConfigurationLoading] = useState(true);
    const [modelConfigurationError, setModelConfigurationError] = useState<string | null>(null);
    const hasFetchedModelConfiguration = useRef(false);

    const workflowId = workflow.id;

    const initialFlow = useMemo(
        () => ({
            nodes: workflow.workflow_definition.nodes as FlowNode[],
            edges: workflow.workflow_definition.edges as FlowEdge[],
            viewport: { x: 0, y: 0, zoom: 0 },
        }),
        [workflow],
    );

    const initialTemplateContextVariables = useMemo(
        () => (workflow.template_context_variables as Record<string, string>) || {},
        [workflow],
    );

    const {
        workflowName,
        configurationState,
        configurationLoadError,
        reloadConfiguration,
        defaultCallDispositions,
        textChatInactivityTimeoutConstraints,
        widgetTextDefaults,
        templateContextVariables,
        saveWorkflowConfigurations,
        saveTemplateContextVariables,
    } = useWorkflowState({
        initialWorkflowName: workflow.name,
        workflowId,
        initialFlow,
        initialTemplateContextVariables,
        user,
    });

    // Each successful load hands down a fresh `configurationState` object. The
    // sections seed their local state from it once, so bump a key to remount
    // them on every load instead of leaving stale values on screen.
    const [configurationVersion, setConfigurationVersion] = useState(0);
    const lastConfigurationState = useRef(configurationState);
    useEffect(() => {
        if (lastConfigurationState.current === configurationState) return;
        lastConfigurationState.current = configurationState;
        setConfigurationVersion((version) => version + 1);
    }, [configurationState]);

    useEffect(() => {
        if (hasFetchedModelConfiguration.current) return;
        hasFetchedModelConfiguration.current = true;

        const loadModelConfiguration = async () => {
            setModelConfigurationLoading(true);
            setModelConfigurationError(null);
            const [defaultsResult, configurationResult, pricingResult] = await Promise.all([
                getModelConfigurationV2DefaultsApiV1OrganizationsModelConfigurationsV2DefaultsGet(),
                getModelConfigurationV2ApiV1OrganizationsModelConfigurationsV2Get(),
                fetchModelConfigurationPricing(),
            ]);

            if (defaultsResult.error) {
                setModelConfigurationError(detailFromError(defaultsResult.error, "Failed to load model configuration defaults"));
                setModelConfigurationLoading(false);
                return;
            }
            if (configurationResult.error) {
                setModelConfigurationError(detailFromError(configurationResult.error, "Failed to load model configuration"));
                setModelConfigurationLoading(false);
                return;
            }

            setModelConfigurationDefaults(defaultsResult.data as ModelConfigurationDefaultsV2);
            setOrganizationModelConfiguration(configurationResult.data || null);
            setModelConfigurationPricing(pricingResult);
            setModelConfigurationLoading(false);
        };

        loadModelConfiguration();
    }, []);

    // Intersection observer for active sidebar link
    useEffect(() => {
        const ids = NAV_ITEMS.map((n) => n.id);
        const observer = new IntersectionObserver(
            (entries) => {
                for (const entry of entries) {
                    if (entry.isIntersecting) {
                        setActiveSection(entry.target.id);
                        break;
                    }
                }
            },
            { rootMargin: "-20% 0px -60% 0px" },
        );
        ids.forEach((id) => {
            const el = document.getElementById(id);
            if (el) observer.observe(el);
        });
        return () => observer.disconnect();
    }, []);

    return (
        <div className="min-h-screen">
            {/* Sticky header */}
            <header className="sticky top-0 z-10 flex items-center gap-3 border-b bg-background/95 px-6 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => confirmNavigate(() => router.push(`/workflow/${workflowId}`))}
                >
                    <ArrowLeft className="h-4 w-4" />
                </Button>
                <div>
                    <p className="text-xs text-muted-foreground">Workflow Settings</p>
                    <h1 className="text-sm font-semibold">{workflowName || workflow.name}</h1>
                </div>
            </header>

            {/* Main + right nav */}
            <div className="mx-auto flex max-w-5xl gap-8 px-6 py-8">
                {/* Sections */}
                <div className="min-w-0 flex-1 space-y-8">
                    {configurationLoadError && (
                        <Card id="configuration-error">
                            <CardHeader>
                                <CardTitle className="text-base">Settings unavailable</CardTitle>
                                <CardDescription>{configurationLoadError}</CardDescription>
                            </CardHeader>
                            <CardFooter>
                                <Button
                                    type="button"
                                    variant="outline"
                                    onClick={() => { void reloadConfiguration(); }}
                                >
                                    Retry
                                </Button>
                            </CardFooter>
                        </Card>
                    )}
                    {configurationState && (
                        <>
                            {/* General */}
                            <GeneralSection
                                key={configurationVersion}
                                configuration={configurationState}
                                defaultCallDispositions={defaultCallDispositions}
                                workflowName={workflowName || workflow.name}
                                workflowId={workflowId}
                                onSave={saveWorkflowConfigurations}
                            />

                            <WorkflowModelOverridesSection
                                key={`models-${configurationVersion}`}
                                configuration={configurationState}
                                onSave={saveWorkflowConfigurations}
                                modelConfigurationDefaults={modelConfigurationDefaults}
                                organizationModelConfiguration={organizationModelConfiguration}
                                modelConfigurationPricing={modelConfigurationPricing}
                                modelConfigurationLoading={modelConfigurationLoading}
                                modelConfigurationError={modelConfigurationError}
                            />

                            {/* Template Variables */}
                            <TemplateVariablesSection
                                templateContextVariables={templateContextVariables}
                                onSave={saveTemplateContextVariables}
                            />

                            {/* Dictionary */}
                            <DictionarySection
                                key={`dictionary-${configurationVersion}`}
                                configuration={configurationState}
                                onSave={saveWorkflowConfigurations}
                            />

                            {/* Voicemail Detection */}
                            <VoicemailSection
                                key={`voicemail-${configurationVersion}`}
                                configuration={configurationState}
                                onSave={saveWorkflowConfigurations}
                            />

                            {/* Recordings – moved to org-level page */}
                            <Card id="recordings">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-base">
                                        <Mic className="h-4 w-4" />
                                        Recordings
                                    </CardTitle>
                                    <CardDescription>
                                        Recordings are now managed at the organization level and shared across all agents.
                                        Use <code className="rounded bg-muted px-1 text-xs">@</code> in prompt fields to insert them.{" "}
                                        <a href={SETTINGS_DOCUMENTATION_URLS.recordings} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 underline">Learn more <ExternalLink className="h-3 w-3" /></a>
                                    </CardDescription>
                                </CardHeader>
                                <CardFooter className="border-t pt-6">
                                    <Button variant="outline" asChild>
                                        <Link href="/recordings">
                                            Go to Recordings
                                            <ExternalLink className="ml-2 h-4 w-4" />
                                        </Link>
                                    </Button>
                                </CardFooter>
                            </Card>

                            {/* Deployment (dialog trigger) */}
                            <Card id="deployment">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-base">
                                        <Rocket className="h-4 w-4" />
                                        Add to Website
                                    </CardTitle>
                                    <CardDescription>
                                        Configure a widget to add this voice agent to your website.{" "}
                                        <a href={SETTINGS_DOCUMENTATION_URLS.deployment} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 underline">Learn more <ExternalLink className="h-3 w-3" /></a>
                                    </CardDescription>
                                </CardHeader>
                                <CardFooter className="border-t pt-6">
                                    <Button variant="outline" onClick={() => setIsEmbedDialogOpen(true)}>
                                        Configure Widget
                                    </Button>
                                </CardFooter>
                            </Card>

                            {/* Report */}
                            <ReportSection workflowId={workflowId} />

                            {/* Agent UUID */}
                            {workflow.workflow_uuid && (
                                <AgentUuidSection workflowUuid={workflow.workflow_uuid} />
                            )}
                        </>
                    )}
                </div>

                {/* ---- Right-side sticky nav ---- */}
                <nav className="hidden w-44 shrink-0 lg:block">
                    <div className="sticky top-20 space-y-1">
                        <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                            On this page
                        </p>
                        {NAV_ITEMS.map((item) => (
                            <a
                                key={item.id}
                                href={`#${item.id}`}
                                className={`flex items-center gap-1.5 rounded-md px-2 py-1 text-sm transition-colors hover:text-foreground ${
                                    activeSection === item.id
                                        ? "font-medium text-foreground"
                                        : "text-muted-foreground"
                                }`}
                            >
                                {item.label}
                                {dirtySections.has(item.id) && (
                                    <span className="h-1.5 w-1.5 rounded-full bg-orange-500" />
                                )}
                            </a>
                        ))}
                    </div>
                </nav>
            </div>

            {/* Dialogs for complex sections */}
            {configurationState && (
                <EmbedDialog
                    open={isEmbedDialogOpen}
                    onOpenChange={setIsEmbedDialogOpen}
                    workflowId={workflowId}
                    workflowName={workflowName || workflow.name}
                    configuration={configurationState}
                    textChatInactivityTimeoutConstraints={textChatInactivityTimeoutConstraints}
                    widgetTextDefaults={widgetTextDefaults}
                    onSaveWorkflowConfigurations={saveWorkflowConfigurations}
                />
            )}
        </div>
    );
}
