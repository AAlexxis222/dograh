"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import {
  deleteLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsDelete,
  getLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsGet,
  saveLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsPost,
} from "@/client/sdk.gen";
import type { LangfuseCredentialsResponse } from "@/client/types.gen";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { detailFromError } from "@/lib/apiError";
import { useAuth } from "@/lib/auth";
import { refusalHint } from "@/lib/knobRefusal";

// A B0-28 refusal (not an operator, no target organization) says what to do in
// its hint; any other backend error shows its own detail.
function errorMessage(error: unknown, fallback: string): string {
  return refusalHint(error) ?? detailFromError(error, fallback);
}

function TracesPublicToggle({
  checked,
  onCheckedChange,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div className="space-y-2 pt-2 border-t">
      <div className="flex items-center space-x-2">
        <Switch
          id="langfuse-traces-public"
          checked={checked}
          onCheckedChange={onCheckedChange}
        />
        <Label htmlFor="langfuse-traces-public">Make traces publicly viewable</Label>
      </div>
      <p className="text-xs text-muted-foreground">
        Off by default. When on, anyone holding a trace URL can open it without
        logging in to Langfuse — including the full call transcript, prompts and
        tool payloads. Turn it on only if you intend to share trace links outside
        your Langfuse project.
      </p>
    </div>
  );
}

export function TelemetrySection() {
  const { user, loading: authLoading } = useAuth();
  const [credentials, setCredentials] = useState<LangfuseCredentialsResponse>({
    host: "",
    public_key: "",
    secret_key: "",
    project_id: "",
    traces_public: false,
    configured: false,
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const hasFetched = useRef(false);

  useEffect(() => {
    if (authLoading || !user || hasFetched.current) {
      return;
    }
    hasFetched.current = true;
    fetchCredentials();
  }, [authLoading, user]);

  async function fetchCredentials() {
    try {
      const { data } = await getLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsGet();
      if (data) {
        setCredentials(data);
      }
    } catch {
      // No credentials configured yet — that's fine
    } finally {
      setLoading(false);
    }
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    try {
      const { error } = await saveLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsPost({
        body: {
          host: credentials.host ?? "",
          public_key: credentials.public_key ?? "",
          secret_key: credentials.secret_key ?? "",
          project_id: credentials.project_id ?? "",
          traces_public: credentials.traces_public ?? false,
        },
      });
      if (error) {
        toast.error(errorMessage(error, "Failed to save telemetry credentials"));
        return;
      }
      toast.success("Telemetry credentials saved");
      await fetchCredentials();
    } catch {
      toast.error("Failed to save telemetry credentials");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    setSaving(true);
    try {
      const { error } = await deleteLangfuseCredentialsApiV1OrganizationsLangfuseCredentialsDelete();
      if (error) {
        toast.error(errorMessage(error, "Failed to remove telemetry credentials"));
        return;
      }
      setCredentials({
        host: "",
        public_key: "",
        secret_key: "",
        project_id: "",
        traces_public: false,
        configured: false,
      });
      toast.success("Telemetry credentials removed");
    } catch {
      toast.error("Failed to remove telemetry credentials");
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return <p className="text-sm text-muted-foreground">Loading...</p>;
  }

  return (
    <form onSubmit={handleSave} className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Connect your Langfuse project to receive call tracing data.
      </p>
      <div className="space-y-2">
        <Label htmlFor="langfuse-host">Host</Label>
        <Input
          id="langfuse-host"
          placeholder="https://cloud.langfuse.com"
          value={credentials.host}
          onChange={(e) => setCredentials({ ...credentials, host: e.target.value })}
          required
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="langfuse-public-key">Public Key</Label>
        <Input
          id="langfuse-public-key"
          placeholder="pk-lf-..."
          value={credentials.public_key}
          onChange={(e) => setCredentials({ ...credentials, public_key: e.target.value })}
          required
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="langfuse-secret-key">Secret Key</Label>
        <Input
          id="langfuse-secret-key"
          type="password"
          placeholder="sk-lf-..."
          value={credentials.secret_key}
          onChange={(e) => setCredentials({ ...credentials, secret_key: e.target.value })}
          required
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="langfuse-project-id">Project ID</Label>
        <Input
          id="langfuse-project-id"
          placeholder="cm..."
          value={credentials.project_id}
          onChange={(e) => setCredentials({ ...credentials, project_id: e.target.value })}
          required
        />
        <p className="text-xs text-muted-foreground">
          Found in your Langfuse URL: /project/&lt;project-id&gt;/traces. Required to
          build links to your traces.
        </p>
      </div>
      <TracesPublicToggle
        checked={credentials.traces_public ?? false}
        onCheckedChange={(checked) =>
          setCredentials({ ...credentials, traces_public: checked })
        }
      />
      <div className="flex gap-2">
        <Button type="submit" disabled={saving}>
          {saving ? "Saving..." : "Save"}
        </Button>
        {credentials.configured && (
          <Button type="button" variant="destructive" disabled={saving} onClick={handleDelete}>
            Remove
          </Button>
        )}
      </div>
    </form>
  );
}
