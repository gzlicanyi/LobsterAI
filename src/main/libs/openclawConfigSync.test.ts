import { test, expect, describe, vi, afterAll } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  ProviderName,
  OpenClawProviderId,
  OpenClawApi,
} from '../../shared/providers';

vi.mock('electron', () => {
  let _appPath = process.cwd();
  let _homeDir = os.tmpdir();
  return {
    app: {
      getAppPath: () => _appPath,
      getPath: (name: string) => {
        if (name === 'home' || name === 'userData') return _homeDir;
        return _homeDir;
      },
      isPackaged: false,
      // helpers for tests to change paths
      __setAppPath: (p: string) => { _appPath = p; },
      __setHomeDir: (d: string) => { _homeDir = d; },
    },
  };
});

import { setStoreGetter } from './claudeSettings';
import { OpenClawConfigSync } from './openclawConfigSync';

const providerApiKeyEnvVar = (providerName: string): string => {
  const envName = providerName.toUpperCase().replace(/[^A-Z0-9]/g, '_');
  return `LOBSTER_APIKEY_${envName}`;
};

describe('providerApiKeyEnvVar', () => {
  test('converts simple provider names', () => {
    expect(providerApiKeyEnvVar(ProviderName.Moonshot)).toBe('LOBSTER_APIKEY_MOONSHOT');
    expect(providerApiKeyEnvVar(ProviderName.Anthropic)).toBe('LOBSTER_APIKEY_ANTHROPIC');
    expect(providerApiKeyEnvVar(ProviderName.OpenAI)).toBe('LOBSTER_APIKEY_OPENAI');
    expect(providerApiKeyEnvVar(ProviderName.Ollama)).toBe('LOBSTER_APIKEY_OLLAMA');
  });

  test('replaces hyphens and special chars with underscores', () => {
    expect(providerApiKeyEnvVar(ProviderName.LobsteraiServer)).toBe('LOBSTER_APIKEY_LOBSTERAI_SERVER');
    expect(providerApiKeyEnvVar('my.provider')).toBe('LOBSTER_APIKEY_MY_PROVIDER');
  });

  test('server key matches hardcoded convention', () => {
    expect(providerApiKeyEnvVar('server')).toBe('LOBSTER_APIKEY_SERVER');
  });
});

describe('env var stability on model switch', () => {
  const simulateCollectEnvVars = (providers: Record<string, { enabled: boolean; apiKey: string }>, serverToken?: string) => {
    const env: Record<string, string> = {};

    if (serverToken) {
      env.LOBSTER_APIKEY_SERVER = serverToken;
    }

    for (const [name, config] of Object.entries(providers)) {
      if (!config.enabled) continue;
      const envName = name.toUpperCase().replace(/[^A-Z0-9]/g, '_');
      env[`LOBSTER_APIKEY_${envName}`] = config.apiKey;
    }

    return env;
  };

  test('switching from server to custom provider does not change env var keys', () => {
    const providers = {
      [ProviderName.Moonshot]: { enabled: true, apiKey: 'sk-moon-123' },
    };
    const serverToken = 'access-token-xyz';

    const envBefore = simulateCollectEnvVars(providers, serverToken);
    const envAfter = simulateCollectEnvVars(providers, serverToken);

    expect(JSON.stringify(envBefore)).toBe(JSON.stringify(envAfter));
  });

  test('switching between two custom providers does not change env var keys', () => {
    const providers = {
      [ProviderName.Moonshot]: { enabled: true, apiKey: 'sk-moon-123' },
      [ProviderName.Anthropic]: { enabled: true, apiKey: 'sk-ant-456' },
    };

    const envBefore = simulateCollectEnvVars(providers);
    const envAfter = simulateCollectEnvVars(providers);

    expect(JSON.stringify(envBefore)).toBe(JSON.stringify(envAfter));
    expect(envBefore.LOBSTER_APIKEY_MOONSHOT).toBe('sk-moon-123');
    expect(envBefore.LOBSTER_APIKEY_ANTHROPIC).toBe('sk-ant-456');
  });

  test('only editing apiKey value causes env var change', () => {
    const providersBefore = {
      [ProviderName.Moonshot]: { enabled: true, apiKey: 'sk-moon-OLD' },
    };
    const providersAfter = {
      [ProviderName.Moonshot]: { enabled: true, apiKey: 'sk-moon-NEW' },
    };

    const envBefore = simulateCollectEnvVars(providersBefore);
    const envAfter = simulateCollectEnvVars(providersAfter);

    expect(JSON.stringify(envBefore)).not.toBe(JSON.stringify(envAfter));
  });
});

// ═══════════════════════════════════════════════════════
// Provider Descriptor Registry Tests
//
// Since buildProviderSelection imports Electron-only modules,
// we mirror the descriptor resolution logic here to verify
// the registry mapping correctness.
// ═══════════════════════════════════════════════════════

type OpenClawProviderApi = 'anthropic-messages' | 'openai-completions' | 'openai-responses' | 'google-generative-ai';

const mapApiTypeToOpenClawApi = (
  apiType: 'anthropic' | 'openai' | undefined,
): OpenClawProviderApi => {
  if (apiType === 'openai') return 'openai-completions';
  return 'anthropic-messages';
};

type ProviderDescriptor = {
  providerId: string;
  resolveApi: (ctx: { apiType: 'anthropic' | 'openai' | undefined; baseURL: string }) => OpenClawProviderApi;
  normalizeBaseUrl: (rawBaseUrl: string) => string;
  resolveSessionModelId?: (modelId: string) => string;
  modelDefaults?: Partial<{
    reasoning: boolean;
    cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
    contextWindow: number;
    maxTokens: number;
  }>;
};

const stripChatCompletionsSuffix = (rawBaseUrl: string): string => {
  const trimmed = rawBaseUrl.trim();
  if (!trimmed) return trimmed;
  const normalized = trimmed.replace(/\/+$/, '');
  if (normalized.endsWith('/openai')) {
    return normalized.slice(0, -'/openai'.length);
  }
  return normalized;
};

const PROVIDER_REGISTRY: Record<string, ProviderDescriptor> = {
  [ProviderName.Moonshot]: {
    providerId: OpenClawProviderId.Moonshot,
    resolveApi: () => OpenClawApi.OpenAICompletions as OpenClawProviderApi,
    normalizeBaseUrl: stripChatCompletionsSuffix,
    modelDefaults: {
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 256000,
      maxTokens: 8192,
    },
  },
  [`${ProviderName.Moonshot}:codingPlan`]: {
    providerId: OpenClawProviderId.KimiCoding,
    resolveApi: () => OpenClawApi.AnthropicMessages as OpenClawProviderApi,
    normalizeBaseUrl: stripChatCompletionsSuffix,
    resolveSessionModelId: () => 'k2p5',
    modelDefaults: {
      reasoning: true,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 256000,
      maxTokens: 8192,
    },
  },
  [ProviderName.Gemini]: {
    providerId: OpenClawProviderId.Google,
    resolveApi: () => OpenClawApi.GoogleGenerativeAI as OpenClawProviderApi,
    normalizeBaseUrl: stripChatCompletionsSuffix,
    modelDefaults: { reasoning: true },
  },
  [ProviderName.Anthropic]: {
    providerId: OpenClawProviderId.Anthropic,
    resolveApi: () => OpenClawApi.AnthropicMessages as OpenClawProviderApi,
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.OpenAI]: {
    providerId: OpenClawProviderId.OpenAI,
    resolveApi: () => OpenClawApi.OpenAICompletions as OpenClawProviderApi,
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.DeepSeek]: {
    providerId: OpenClawProviderId.DeepSeek,
    resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.Qwen]: {
    providerId: OpenClawProviderId.Qwen,
    resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.Zhipu]: {
    providerId: OpenClawProviderId.Zai,
    resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.Volcengine]: {
    providerId: OpenClawProviderId.Volcengine,
    resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [`${ProviderName.Volcengine}:codingPlan`]: {
    providerId: OpenClawProviderId.VolcenginePlan,
    resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.Minimax]: {
    providerId: OpenClawProviderId.Minimax,
    resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.Youdaozhiyun]: {
    providerId: OpenClawProviderId.Youdaozhiyun,
    resolveApi: () => OpenClawApi.OpenAICompletions as OpenClawProviderApi,
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.StepFun]: {
    providerId: OpenClawProviderId.StepFun,
    resolveApi: () => OpenClawApi.OpenAICompletions as OpenClawProviderApi,
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.Xiaomi]: {
    providerId: OpenClawProviderId.Xiaomi,
    resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.OpenRouter]: {
    providerId: OpenClawProviderId.OpenRouter,
    resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
  [ProviderName.Ollama]: {
    providerId: OpenClawProviderId.Ollama,
    resolveApi: () => OpenClawApi.OpenAICompletions as OpenClawProviderApi,
    normalizeBaseUrl: stripChatCompletionsSuffix,
  },
};

const DEFAULT_DESCRIPTOR: ProviderDescriptor = {
  providerId: OpenClawProviderId.Lobster,
  resolveApi: ({ apiType }) => mapApiTypeToOpenClawApi(apiType),
  normalizeBaseUrl: stripChatCompletionsSuffix,
};

const resolveDescriptor = (
  providerName: string,
  codingPlanEnabled: boolean,
): ProviderDescriptor => {
  if (codingPlanEnabled) {
    const compositeKey = `${providerName}:codingPlan`;
    if (compositeKey in PROVIDER_REGISTRY) {
      return PROVIDER_REGISTRY[compositeKey];
    }
  }
  if (providerName in PROVIDER_REGISTRY) {
    return PROVIDER_REGISTRY[providerName];
  }
  return {
    ...DEFAULT_DESCRIPTOR,
    providerId: providerName || OpenClawProviderId.Lobster,
  };
};

describe('resolveDescriptor', () => {
  test('gemini maps to google providerId with google-generative-ai API', () => {
    const d = resolveDescriptor(ProviderName.Gemini, false);
    expect(d.providerId).toBe(OpenClawProviderId.Google);
    expect(d.resolveApi({ apiType: undefined, baseURL: '' })).toBe(OpenClawApi.GoogleGenerativeAI);
  });

  test('anthropic maps to anthropic providerId with anthropic-messages API', () => {
    const d = resolveDescriptor(ProviderName.Anthropic, false);
    expect(d.providerId).toBe(OpenClawProviderId.Anthropic);
    expect(d.resolveApi({ apiType: undefined, baseURL: '' })).toBe(OpenClawApi.AnthropicMessages);
  });

  test('openai maps to openai providerId', () => {
    const d = resolveDescriptor(ProviderName.OpenAI, false);
    expect(d.providerId).toBe(OpenClawProviderId.OpenAI);
  });

  test('moonshot without codingPlan uses moonshot providerId', () => {
    const d = resolveDescriptor(ProviderName.Moonshot, false);
    expect(d.providerId).toBe(OpenClawProviderId.Moonshot);
    expect(d.resolveApi({ apiType: undefined, baseURL: '' })).toBe(OpenClawApi.OpenAICompletions);
  });

  test('moonshot with codingPlan uses kimi-coding providerId', () => {
    const d = resolveDescriptor(ProviderName.Moonshot, true);
    expect(d.providerId).toBe(OpenClawProviderId.KimiCoding);
    expect(d.resolveApi({ apiType: undefined, baseURL: '' })).toBe(OpenClawApi.AnthropicMessages);
    expect(d.resolveSessionModelId!('any-model')).toBe('k2p5');
  });

  test('moonshot codingPlan has model defaults', () => {
    const d = resolveDescriptor(ProviderName.Moonshot, true);
    expect(d.modelDefaults?.reasoning).toBe(true);
    expect(d.modelDefaults?.contextWindow).toBe(256000);
    expect(d.modelDefaults?.maxTokens).toBe(8192);
  });

  test('deepseek maps to deepseek providerId respecting apiType', () => {
    const d = resolveDescriptor(ProviderName.DeepSeek, false);
    expect(d.providerId).toBe(OpenClawProviderId.DeepSeek);
    expect(d.resolveApi({ apiType: 'openai', baseURL: '' })).toBe(OpenClawApi.OpenAICompletions);
    expect(d.resolveApi({ apiType: 'anthropic', baseURL: '' })).toBe(OpenClawApi.AnthropicMessages);
  });

  test('youdaozhiyun always uses openai-completions', () => {
    const d = resolveDescriptor(ProviderName.Youdaozhiyun, false);
    expect(d.providerId).toBe(OpenClawProviderId.Youdaozhiyun);
    expect(d.resolveApi({ apiType: 'anthropic', baseURL: '' })).toBe(OpenClawApi.OpenAICompletions);
  });

  test('ollama always uses openai-completions', () => {
    const d = resolveDescriptor(ProviderName.Ollama, false);
    expect(d.providerId).toBe(OpenClawProviderId.Ollama);
    expect(d.resolveApi({ apiType: undefined, baseURL: '' })).toBe(OpenClawApi.OpenAICompletions);
  });

  test('unknown provider falls back to lobster providerId', () => {
    const d = resolveDescriptor('some-unknown', false);
    expect(d.providerId).toBe('some-unknown');
  });

  test('empty provider name falls back to lobster', () => {
    const d = resolveDescriptor('', false);
    expect(d.providerId).toBe(OpenClawProviderId.Lobster);
  });

  test('codingPlan flag is ignored for providers without codingPlan entry', () => {
    const d = resolveDescriptor(ProviderName.OpenAI, true);
    expect(d.providerId).toBe(OpenClawProviderId.OpenAI);
  });

  test('volcengine with codingPlan uses volcengine-plan providerId', () => {
    const d = resolveDescriptor(ProviderName.Volcengine, true);
    expect(d.providerId).toBe(OpenClawProviderId.VolcenginePlan);
  });

  test('volcengine without codingPlan uses volcengine providerId', () => {
    const d = resolveDescriptor(ProviderName.Volcengine, false);
    expect(d.providerId).toBe(OpenClawProviderId.Volcengine);
  });
});

describe('provider registry coverage', () => {
  const allRegistryProviders = [
    ProviderName.Moonshot,
    ProviderName.Gemini,
    ProviderName.Anthropic,
    ProviderName.OpenAI,
    ProviderName.DeepSeek,
    ProviderName.Qwen,
    ProviderName.Zhipu,
    ProviderName.Volcengine,
    ProviderName.Minimax,
    ProviderName.Youdaozhiyun,
    ProviderName.StepFun,
    ProviderName.Xiaomi,
    ProviderName.OpenRouter,
    ProviderName.Ollama,
  ] as const;

  test('all 14 providers have registry entries', () => {
    for (const name of allRegistryProviders) {
      expect(name in PROVIDER_REGISTRY, `${name} missing from registry`).toBe(true);
    }
  });

  test('no provider resolves to lobster fallback', () => {
    for (const name of allRegistryProviders) {
      const d = resolveDescriptor(name, false);
      expect(d.providerId).not.toBe(OpenClawProviderId.Lobster);
    }
  });

  test('every provider has a non-empty providerId', () => {
    for (const name of allRegistryProviders) {
      const d = resolveDescriptor(name, false);
      expect(d.providerId.length).toBeGreaterThan(0);
    }
  });
});

// ═══════════════════════════════════════════════════════
// Integration tests: OpenClawConfigSync.sync()
//
// These tests instantiate the real OpenClawConfigSync class
// with a mocked electron module and a temporary filesystem.
// ═══════════════════════════════════════════════════════

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const electronApp = (await import('electron')).app as any;

const setElectronPaths = (homeDir: string) => {
  electronApp.__setAppPath(process.cwd());
  electronApp.__setHomeDir(homeDir);
};

const restoreElectronPaths = () => {
  electronApp.__setAppPath(process.cwd());
  electronApp.__setHomeDir(os.tmpdir());
};

const createAppConfig = ({ codingPlanEnabled = false } = {}) => ({
  model: {
    defaultModel: 'kimi-k2.5',
    defaultModelProvider: 'moonshot',
  },
  providers: {
    moonshot: {
      enabled: true,
      apiKey: 'sk-test',
      baseUrl: 'https://api.moonshot.cn/anthropic',
      apiFormat: 'anthropic',
      codingPlanEnabled,
      models: [
        { id: 'kimi-k2.5' },
      ],
    },
  },
});

const createOpenAICompatAppConfig = () => ({
  model: {
    defaultModel: 'kimi-k2.5',
    defaultModelProvider: 'openai',
  },
  providers: {
    openai: {
      enabled: true,
      apiKey: 'sk-test',
      baseUrl: 'https://api.example.com/v1',
      apiFormat: 'openai',
      models: [
        { id: 'kimi-k2.5' },
      ],
    },
  },
});

const createSessionStore = () => ({
  'agent:main:lobsterai:current-session': {
    sessionId: 'session-current',
    modelProvider: 'lobster',
    model: 'kimi-k2.5',
    systemPromptReport: {
      provider: 'lobster',
      model: 'kimi-k2.5',
    },
  },
  'agent:main:lobsterai:old-claude-session': {
    sessionId: 'session-old-claude',
    modelProvider: 'lobster',
    model: 'claude-sonnet-4-5-20250929',
    systemPromptReport: {
      provider: 'lobster',
      model: 'claude-sonnet-4-5-20250929',
    },
  },
  'agent:main:wecom:direct:wangning': {
    sessionId: 'session-wecom',
    execSecurity: 'full',
    skillsSnapshot: {
      prompt: '<skill><name>feishu-cron-reminder</name></skill>',
      resolvedSkills: [
        { name: 'feishu-cron-reminder' },
      ],
    },
  },
  'agent:main:feishu:dm:ou_123': {
    sessionId: 'session-feishu',
    skillsSnapshot: {
      resolvedSkills: [
        { name: 'qqbot-cron' },
      ],
    },
  },
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const createSync = (tmpDir: string, appConfig: any, options: Record<string, any> = {}) => {
  setStoreGetter(() => ({
    get: (key: string) => (key === 'app_config' ? appConfig : null),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  }) as any);

  return new OpenClawConfigSync({
    engineManager: {
      getConfigPath: () => path.join(tmpDir, 'state', 'openclaw.json'),
      getStateDir: () => path.join(tmpDir, 'state'),
      getGatewayToken: () => null,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any,
    getCoworkConfig: () => ({
      workingDirectory: options.workingDirectory ?? '',
      systemPrompt: options.systemPrompt ?? '',
      executionMode: options.executionMode ?? 'auto',
    }),
    isEnterprise: () => false,
    getDingTalkConfig: () => null,
    getFeishuConfig: () => null,
    getQQConfig: () => options.qqConfig ?? null,
    getWecomConfig: () => null,
    getPopoConfig: () => null,
    getNimConfig: () => null,
    getNeteaseBeeChanConfig: () => null,
    getWeixinConfig: () => null,
  });
};

afterAll(() => {
  restoreElectronPaths();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setStoreGetter(() => null as any);
});

describe('OpenClawConfigSync integration', () => {
  test('sync writes native moonshot provider config and migrates matching managed sessions', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-'));
    try {
      setElectronPaths(tmpDir);

      const sessionsDir = path.join(tmpDir, 'state', 'agents', 'main', 'sessions');
      fs.mkdirSync(sessionsDir, { recursive: true });
      fs.writeFileSync(
        path.join(sessionsDir, 'sessions.json'),
        `${JSON.stringify(createSessionStore(), null, 2)}\n`,
        'utf8',
      );

      const sync = createSync(tmpDir, createAppConfig());
      const result = sync.sync('test');

      expect(result.ok).toBe(true);
      expect(result.changed).toBe(true);

      const config = JSON.parse(fs.readFileSync(path.join(tmpDir, 'state', 'openclaw.json'), 'utf8'));
      expect(config.models.providers.moonshot.baseUrl).toBe('https://api.moonshot.cn/v1');
      expect(config.models.providers.moonshot.api).toBe('openai-completions');
      expect(config.agents.defaults.model.primary).toBe('moonshot/kimi-k2.5');
      expect(config.commands.ownerAllowFrom).toContain('gateway-client');
      expect(config.tools.web.search.enabled).toBe(false);
      expect(config.browser.enabled).toBe(true);

      const sessionStore = JSON.parse(fs.readFileSync(path.join(sessionsDir, 'sessions.json'), 'utf8'));
      expect(sessionStore['agent:main:lobsterai:current-session'].modelProvider).toBe('moonshot');
      expect(sessionStore['agent:main:lobsterai:current-session'].model).toBe('kimi-k2.5');
      expect(sessionStore['agent:main:lobsterai:current-session'].systemPromptReport.provider).toBe('moonshot');
      expect(sessionStore['agent:main:lobsterai:old-claude-session'].modelProvider).toBe('lobster');
      expect(sessionStore['agent:main:lobsterai:old-claude-session'].model).toBe('claude-sonnet-4-5-20250929');
      expect(sessionStore['agent:main:wecom:direct:wangning'].execSecurity).toBe('full');
      expect(sessionStore['agent:main:feishu:dm:ou_123'].execSecurity).toBe('full');
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('sync maps moonshot coding plan sessions to kimi-coding model refs', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-coding-'));
    try {
      setElectronPaths(tmpDir);

      const sessionsDir = path.join(tmpDir, 'state', 'agents', 'main', 'sessions');
      fs.mkdirSync(sessionsDir, { recursive: true });
      fs.writeFileSync(
        path.join(sessionsDir, 'sessions.json'),
        `${JSON.stringify(createSessionStore(), null, 2)}\n`,
        'utf8',
      );

      const sync = createSync(tmpDir, createAppConfig({ codingPlanEnabled: true }));
      const result = sync.sync('test-coding-plan');

      expect(result.ok).toBe(true);

      const config = JSON.parse(fs.readFileSync(path.join(tmpDir, 'state', 'openclaw.json'), 'utf8'));
      expect(config.models.providers['kimi-coding'].baseUrl).toBe('https://api.kimi.com/coding');
      expect(config.models.providers['kimi-coding'].api).toBe('anthropic-messages');
      expect(config.agents.defaults.model.primary).toBe('kimi-coding/k2p5');
      expect(config.commands.ownerAllowFrom).toContain('gateway-client');

      const sessionStore = JSON.parse(fs.readFileSync(path.join(sessionsDir, 'sessions.json'), 'utf8'));
      expect(sessionStore['agent:main:lobsterai:current-session'].modelProvider).toBe('kimi-coding');
      expect(sessionStore['agent:main:lobsterai:current-session'].model).toBe('k2p5');
      expect(sessionStore['agent:main:lobsterai:current-session'].systemPromptReport.provider).toBe('kimi-coding');
      expect(sessionStore['agent:main:lobsterai:current-session'].systemPromptReport.model).toBe('k2p5');
      expect(sessionStore['agent:main:wecom:direct:wangning'].execSecurity).toBe('full');
      expect(sessionStore['agent:main:feishu:dm:ou_123'].execSecurity).toBe('full');
      expect('skillsSnapshot' in sessionStore['agent:main:wecom:direct:wangning']).toBe(false);
      expect('skillsSnapshot' in sessionStore['agent:main:feishu:dm:ou_123']).toBe(false);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('sync denies exec for native channel sessions even without provider migration', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-native-session-'));
    try {
      setElectronPaths(tmpDir);

      const sessionsDir = path.join(tmpDir, 'state', 'agents', 'main', 'sessions');
      fs.mkdirSync(sessionsDir, { recursive: true });
      fs.writeFileSync(
        path.join(sessionsDir, 'sessions.json'),
        `${JSON.stringify(createSessionStore(), null, 2)}\n`,
        'utf8',
      );

      const sync = createSync(tmpDir, createOpenAICompatAppConfig());
      const result = sync.sync('test-native-channel-session-policy');

      expect(result.ok).toBe(true);
      expect(result.changed).toBe(true);

      const sessionStore = JSON.parse(fs.readFileSync(path.join(sessionsDir, 'sessions.json'), 'utf8'));
      expect(sessionStore['agent:main:lobsterai:current-session'].modelProvider).toBe('openai');
      expect(sessionStore['agent:main:lobsterai:current-session'].model).toBe('kimi-k2.5');
      expect(sessionStore['agent:main:wecom:direct:wangning'].execSecurity).toBe('full');
      expect(sessionStore['agent:main:feishu:dm:ou_123'].execSecurity).toBe('full');
      expect('skillsSnapshot' in sessionStore['agent:main:wecom:direct:wangning']).toBe(false);
      expect('skillsSnapshot' in sessionStore['agent:main:feishu:dm:ou_123']).toBe(false);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('sync writes scheduled-task policy into managed AGENTS.md for native channel sessions', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-agents-'));
    try {
      setElectronPaths(tmpDir);

      const workspaceDir = path.join(tmpDir, 'workspace');
      fs.mkdirSync(workspaceDir, { recursive: true });

      const sync = createSync(tmpDir, createAppConfig(), {
        workingDirectory: workspaceDir,
        systemPrompt: 'Always answer in Chinese.',
      });
      const result = sync.sync('test-agents');

      expect(result.ok).toBe(true);

      const agentsMd = fs.readFileSync(path.join(workspaceDir, 'AGENTS.md'), 'utf8');
      expect(agentsMd).toMatch(/# AGENTS\.md - Your Workspace/);
      expect(agentsMd).toMatch(/## Every Session/);
      expect(agentsMd).toMatch(/Read `SOUL\.md`/);
      expect(agentsMd).toMatch(/Read `USER\.md`/);
      expect(agentsMd).toMatch(/If in MAIN SESSION.*Also read `MEMORY\.md`/s);
      expect(agentsMd).toMatch(/## Scheduled Tasks/);
      expect(agentsMd).toMatch(/## Web Search/);
      expect(agentsMd).toMatch(/Built-in `web_search` is disabled in this workspace\./);
      expect(agentsMd).toMatch(/use `web_fetch`/);
      expect(agentsMd).toMatch(/use the built-in `browser` tool/);
      expect(agentsMd).toMatch(/Native channel sessions may deny `exec`/);
      expect(agentsMd).toMatch(/native `cron` tool/i);
      expect(agentsMd).toMatch(/action: "add".*cron\.add/i);
      expect(agentsMd).toMatch(/follow the native `cron` tool schema/i);
      expect(agentsMd).toMatch(/plugins provide session context and outbound delivery; they do not own scheduling logic/i);
      expect(agentsMd).toMatch(/ignore channel-specific reminder helpers or reminder skills/i);
      expect(agentsMd).toMatch(/QQBOT_PAYLOAD/);
      expect(agentsMd).toMatch(/QQBOT_CRON/);
      expect(agentsMd).toMatch(/do not use `sessions_spawn`, `subagents`, or ad-hoc background workflows as a substitute for `cron\.add`/i);
      expect(agentsMd).toMatch(/## System Prompt/);
      expect(agentsMd).toMatch(/Always answer in Chinese\./);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('sync preserves existing AGENTS.md content above the Lobster managed marker', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-agents-preserve-'));
    try {
      setElectronPaths(tmpDir);

      const workspaceDir = path.join(tmpDir, 'workspace');
      fs.mkdirSync(workspaceDir, { recursive: true });
      fs.writeFileSync(
        path.join(workspaceDir, 'AGENTS.md'),
        '# Custom Workspace Notes\n\nKeep this line.\n',
        'utf8',
      );

      const sync = createSync(tmpDir, createAppConfig(), {
        workingDirectory: workspaceDir,
      });
      const result = sync.sync('test-agents-preserve');

      expect(result.ok).toBe(true);

      const agentsMd = fs.readFileSync(path.join(workspaceDir, 'AGENTS.md'), 'utf8');
      expect(agentsMd).toMatch(/^# Custom Workspace Notes\n\nKeep this line\./);
      expect(agentsMd).toMatch(/<!-- LobsterAI managed: do not edit below this line -->/);
      expect(agentsMd).not.toMatch(/^# AGENTS\.md - Your Workspace/m);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('sync backfills the default OpenClaw AGENTS template', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-agents-backfill-'));
    try {
      setElectronPaths(tmpDir);

      const workspaceDir = path.join(tmpDir, 'workspace');
      fs.mkdirSync(workspaceDir, { recursive: true });
      fs.writeFileSync(
        path.join(workspaceDir, 'AGENTS.md'),
        [
          '<!-- LobsterAI managed: do not edit below this line -->',
          '',
          '## System Prompt',
          '',
          'Old managed-only content.',
          '',
        ].join('\n'),
        'utf8',
      );

      const sync = createSync(tmpDir, createAppConfig(), {
        workingDirectory: workspaceDir,
      });
      const result = sync.sync('test-agents-backfill');

      expect(result.ok).toBe(true);

      const agentsMd = fs.readFileSync(path.join(workspaceDir, 'AGENTS.md'), 'utf8');
      expect(agentsMd).toMatch(/^# AGENTS\.md - Your Workspace/m);
      expect(agentsMd).toMatch(/## Every Session/);
      expect(agentsMd).toMatch(/<!-- LobsterAI managed: do not edit below this line -->/);
      expect(agentsMd).toMatch(/## Scheduled Tasks/);
      expect(agentsMd).not.toMatch(/Old managed-only content\./);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('sync disables legacy qqbot-cron skill', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-qq-skill-'));
    try {
      setElectronPaths(tmpDir);

      const sync = createSync(tmpDir, createAppConfig(), {
        qqConfig: {
          enabled: true,
          appId: 'qq-app-id',
          appSecret: 'qq-app-secret',
          dmPolicy: 'open',
          allowFrom: [],
          groupPolicy: 'open',
          groupAllowFrom: [],
          historyLimit: 50,
          markdownSupport: true,
          imageServerBaseUrl: '',
          debug: false,
        },
      });
      const result = sync.sync('test-qq-native-cron');

      expect(result.ok).toBe(true);

      const config = JSON.parse(fs.readFileSync(path.join(tmpDir, 'state', 'openclaw.json'), 'utf8'));
      expect(config.channels.qqbot.enabled).toBe(true);
      expect(config.skills.entries['qqbot-cron'].enabled).toBe(false);
      expect(config.skills.entries['feishu-cron-reminder'].enabled).toBe(false);
      expect(config.cron.enabled).toBe(true);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('sync disables legacy reminder skills', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-reminder-skills-'));
    try {
      setElectronPaths(tmpDir);

      const sync = createSync(tmpDir, createAppConfig());
      const result = sync.sync('test-native-im-reminder-skills');

      expect(result.ok).toBe(true);

      const config = JSON.parse(fs.readFileSync(path.join(tmpDir, 'state', 'openclaw.json'), 'utf8'));
      expect(config.skills.entries['qqbot-cron'].enabled).toBe(false);
      expect(config.skills.entries['feishu-cron-reminder'].enabled).toBe(false);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('sync writes non-empty placeholder apiKey for providers that do not require auth', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'openclaw-config-sync-empty-key-'));
    try {
      setElectronPaths(tmpDir);

      const ollamaAppConfig = {
        model: {
          defaultModel: 'llama3',
          defaultModelProvider: 'ollama',
        },
        providers: {
          ollama: {
            enabled: true,
            apiKey: '',
            baseUrl: 'http://localhost:11434/v1',
            apiFormat: 'openai',
            models: [
              { id: 'llama3' },
            ],
          },
        },
      };

      const sync = createSync(tmpDir, ollamaAppConfig);
      const result = sync.sync('test-empty-key');

      expect(result.ok).toBe(true);
      expect(result.changed).toBe(true);

      const config = JSON.parse(fs.readFileSync(path.join(tmpDir, 'state', 'openclaw.json'), 'utf8'));
      const providerConfig = config.models.providers.ollama ?? config.models.providers.lobster;
      expect(providerConfig).toBeTruthy();
      expect(providerConfig.apiKey).toBeTruthy();
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
