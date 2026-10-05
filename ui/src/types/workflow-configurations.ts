import type {
    AmbientNoiseConfigurationDefaults,
    CallDispositionOption as GeneratedCallDispositionOption,
    OrganizationAiModelConfigurationV2,
    WorkflowConfigurationDefaults as GeneratedWorkflowConfigurationDefaults,
    WorkflowEffectiveConfigurationResponse,
} from "@/client/types.gen";

export type AmbientNoiseConfiguration = Omit<
    AmbientNoiseConfigurationDefaults,
    "enabled" | "volume"
> & {
    enabled: boolean;
    volume: number;
    storage_key?: string;
    storage_backend?: string;
    original_filename?: string;
};

export type TurnStopStrategy = NonNullable<GeneratedWorkflowConfigurationDefaults["turn_stop_strategy"]>;
export type TurnStartStrategy = NonNullable<GeneratedWorkflowConfigurationDefaults["turn_start_strategy"]>;
export const DEFAULT_TURN_START_STRATEGY: TurnStartStrategy = 'default';
export const DEFAULT_TURN_START_MIN_WORDS = 3; // fork default (Alexis), not upstream #832's 2

// "provisional_vad" was retired. Definitions saved before then still carry it,
// so map it onto the option the backend now resolves such a value to ("default",
// VOZ-BUG-18), rather than handing the select a value it has no entry for.
export function coerceTurnStartStrategy(value: string): TurnStartStrategy {
    return TURN_START_STRATEGY_OPTIONS.some(o => o.value === value)
        ? (value as TurnStartStrategy)
        : DEFAULT_TURN_START_STRATEGY;
}

export const TURN_START_STRATEGY_OPTIONS: Array<{
    value: TurnStartStrategy;
    label: string;
    description: string;
}> = [
    {
        value: 'default',
        label: 'Voice activity',
        description: 'Interrupt when the STT provider or local voice activity detection signals speech.',
    },
    {
        value: 'min_words',
        label: 'Minimum words',
        description: 'Wait for a minimum number of transcribed words before interrupting bot speech.',
    },
];

export interface AnswerMessage {
    text?: string;
    recording_id?: string;
    recording_pk?: number;
}

export interface AnswerSupervisorSettings {
    listening_window_ms?: number;
    human_utterance_max_ms?: number;
    machine_utterance_cap_ms?: number;
    classify_budget_ms?: number;
    screening_wait_ms?: number;
    max_screening_rearms?: number;
    voicemail_action?: 'hangup' | 'leave_message';
    voicemail_message?: AnswerMessage;
    screening_message?: AnswerMessage;
}

export interface VoicemailDetectionConfiguration extends AnswerSupervisorSettings {
    enabled: boolean;
    use_workflow_llm: boolean;
    provider?: string;
    model?: string;
    api_key?: string;
    system_prompt?: string;  // Overrides the built-in classifier instructions
}

export const DEFAULT_VOICEMAIL_DETECTION_CONFIGURATION: VoicemailDetectionConfiguration = {
    enabled: false,
    use_workflow_llm: true,
    voicemail_action: 'hangup',
};

export interface TranscriptConfiguration {
    include_end_timestamps: boolean;
}

export interface ExternalPBXFieldMapping {
    context_path: string;
    destination_field: string;
}

export type CallDispositionOption = GeneratedCallDispositionOption;

export const DEFAULT_TRANSCRIPT_CONFIGURATION: TranscriptConfiguration = {
    include_end_timestamps: false,
};

export interface ModelOverrides {
    llm?: {
        provider?: string;
        model?: string;
        api_key?: string;
        [key: string]: unknown;
    };
    tts?: {
        provider?: string;
        model?: string;
        voice?: string;
        api_key?: string;
        [key: string]: unknown;
    };
    stt?: {
        provider?: string;
        model?: string;
        api_key?: string;
        [key: string]: unknown;
    };
    realtime?: {
        provider?: string;
        model?: string;
        voice?: string;
        api_key?: string;
        [key: string]: unknown;
    };
    is_realtime?: boolean;
}

type WorkflowConfigurationBase = Omit<
    GeneratedWorkflowConfigurationDefaults,
    | "ambient_noise_configuration"
    | "max_call_duration"
    | "max_user_idle_timeout"
    | "smart_turn_stop_secs"
    | "turn_start_strategy"
    | "turn_start_min_words"
    | "turn_stop_strategy"
    | "dictionary"
    | "context_compaction_enabled"
    | "tts_cache_enabled"
    | "call_dispositions"
    | "text_chat_inactivity_timeout_seconds"
    | "external_pbx_field_mappings"
    | "external_pbx_lead_headers"
    | "voicemail_detection"
    | "transcript_configuration"
    | "model_overrides"
    | "model_configuration_v2_override"
>;

export type WorkflowConfigurations = WorkflowConfigurationBase & {
    ambient_noise_configuration: AmbientNoiseConfiguration;
    max_call_duration: number;  // Maximum call duration in seconds
    max_user_idle_timeout: number;  // Maximum user idle time in seconds
    smart_turn_stop_secs: number;  // Timeout in seconds for incomplete turn detection
    turn_start_strategy: TurnStartStrategy;  // Strategy for detecting start of user turn/interruption
    turn_start_min_words: number;  // Minimum transcribed words required for minimum-word interruptions
    turn_stop_strategy: TurnStopStrategy;  // Strategy for detecting end of user turn
    dictionary?: string;  // Comma-separated words for voice agent to listen for
    voicemail_detection?: VoicemailDetectionConfiguration;
    transcript_configuration: TranscriptConfiguration;
    context_compaction_enabled: boolean;  // Summarize context on node transitions to remove stale tool calls
    tts_cache_enabled: boolean;
    call_dispositions: CallDispositionOption[];  // Allowed terminal business outcomes
    text_chat_inactivity_timeout_seconds?: number;  // End inactive text chats after this many seconds
    external_pbx_field_mappings: ExternalPBXFieldMapping[];
    external_pbx_lead_headers: string[];  // Extra lead fields to capture from the inbound INVITE
    model_overrides?: ModelOverrides;  // Per-workflow model configuration overrides
    model_configuration_v2_override?: OrganizationAiModelConfigurationV2;  // Full v2 model configuration override
    [key: string]: unknown;  // Allow additional properties for future configurations
};

export type SparseWorkflowConfigurations = Record<string, unknown>;

export interface WorkflowConfigurationState {
    effective: WorkflowConfigurations;        // materialised, what runs
    own: SparseWorkflowConfigurations;        // exactly what the workflow stores (masked secrets)
    base: WorkflowConfigurations;             // schema <- organization: what an absent leaf inherits
    warnings: string[];
}

/**
 * The API resolves schema <- organization <- workflow (cascade.py) and returns
 * the three layers; the UI never merges. `effective` and `base` are
 * materialised documents, so the cast mirrors the one the editor already made
 * on `WorkflowResponse.workflow_configurations`.
 */
export function resolveWorkflowConfigurations(
    response: WorkflowEffectiveConfigurationResponse,
): WorkflowConfigurationState {
    return {
        effective: response.effective as WorkflowConfigurations,
        own: response.own ?? {},
        base: response.base as WorkflowConfigurations,
        warnings: response.warnings ?? [],
    };
}
