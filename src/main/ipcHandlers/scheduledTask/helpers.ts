import { PlatformRegistry } from '../../../shared/platform';

export interface ScheduledTaskHelperDeps {
  getIMGatewayManager: () => {
    getConfig: () => Record<string, unknown> | null;
  } | null;
}

let deps: ScheduledTaskHelperDeps | null = null;

export function initScheduledTaskHelpers(d: ScheduledTaskHelperDeps): void {
  deps = d;
}

const MULTI_INSTANCE_CONFIG_KEYS = new Set(['dingtalk', 'feishu', 'qq']);

function isConfigKeyEnabled(key: string, value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;

  if (MULTI_INSTANCE_CONFIG_KEYS.has(key)) {
    const instances = (value as { instances?: unknown[] }).instances;
    if (!Array.isArray(instances) || instances.length === 0) return false;
    return instances.some(
      (inst) => inst && typeof inst === 'object' && (inst as { enabled?: boolean }).enabled,
    );
  }

  return (value as { enabled?: boolean }).enabled === true;
}

export function listScheduledTaskChannels(): Array<{ value: string; label: string }> {
  const manager = deps?.getIMGatewayManager();
  const config = manager?.getConfig();
  if (!config) {
    return [...PlatformRegistry.channelOptions()];
  }

  const configRecord = config as unknown as Record<string, unknown>;

  const enabledPlatforms = new Set<string>();
  for (const [key, value] of Object.entries(configRecord)) {
    if (isConfigKeyEnabled(key, value)) {
      enabledPlatforms.add(key);
    }
  }

  return PlatformRegistry.channelOptions().filter((option) => {
    const platform = PlatformRegistry.platformOfChannel(option.value);
    return platform !== undefined && enabledPlatforms.has(platform);
  });
}
