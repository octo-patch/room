import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { AgentExecutionOptions } from '../agent-executor'

vi.mock('../claude-code', () => ({ executeClaudeCode: vi.fn() }))

import { compressSession, executeAgent } from '../agent-executor'

const mockFetch = vi.fn()
const toolDefs: NonNullable<AgentExecutionOptions['toolDefs']> = [{
  type: 'function' as const,
  function: { name: 'inspect', description: 'Inspect state', parameters: { type: 'object', properties: {} } }
}]
const protocols = ['minimax', 'minimax-anthropic'] as const
const regions = [undefined, 'cn', 'cn_zh'] as const
const models = [undefined, 'MiniMax-M3.1-Flash', 'MiniMax-M3', 'MiniMax-M2.7', 'custom-model'] as const

function reply(prefix: string): Record<string, unknown> {
  return prefix === 'minimax'
    ? { choices: [{ message: { role: 'assistant', content: 'Done' } }] }
    : { content: [{ type: 'text', text: 'Done' }], stop_reason: 'end_turn' }
}

function endpoint(prefix: string, region: string | undefined): string {
  const host = region ? 'api.minimaxi.com' : 'api.minimax.io'
  return `https://${host}${prefix === 'minimax' ? '/v1/chat/completions' : '/anthropic/v1/messages'}`
}

function payload(call = 0): Record<string, any> {
  return JSON.parse(mockFetch.mock.calls[call][1].body)
}

beforeEach(() => {
  mockFetch.mockReset()
  vi.stubGlobal('fetch', mockFetch)
  vi.stubEnv('MINIMAX_API_KEY', '')
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe.each(protocols)('%s request boundary', (prefix) => {
  describe.each(regions)('region %s', (region) => {
    it.each(models)('uses Flash by default and preserves explicit model %s', async (model) => {
      vi.stubEnv('MINIMAX_REGION', region)
      mockFetch.mockResolvedValue({ ok: true, json: async () => reply(prefix) })
      const result = await executeAgent({
        model: model ? `${prefix}:${model}` : prefix,
        apiKey: 'fixture-key',
        prompt: 'Hello',
        systemPrompt: 'Be helpful'
      })
      expect(result.exitCode).toBe(0)
      expect(mockFetch).toHaveBeenCalledOnce()
      expect(mockFetch.mock.calls[0][0]).toBe(endpoint(prefix, region))
      expect(mockFetch.mock.calls[0][1].headers).toMatchObject(prefix === 'minimax'
        ? { Authorization: 'Bearer fixture-key' }
        : { 'x-api-key': 'fixture-key' })
      expect(payload().model).toBe(model ?? 'MiniMax-M3.1-Flash')
      expect(payload().messages).toContainEqual({ role: 'user', content: 'Hello' })
      // Leave provider thinking/effort defaults intact; Flash rejects disabled/none.
      for (const field of ['thinking', 'reasoning_effort', 'output_config', 'service_tier', 'reasoning_split']) {
        expect(payload()).not.toHaveProperty(field)
      }
      expect(result.output).toBe('Done')
    })

    it('uses the same Flash default for session compression', async () => {
      vi.stubEnv('MINIMAX_REGION', region)
      mockFetch.mockResolvedValue({ ok: true, json: async () => reply(prefix) })
      expect(await compressSession(prefix, 'fixture-key', [{ role: 'user', content: 'Prior work' }])).toBe('Done')
      expect(mockFetch.mock.calls[0][0]).toBe(endpoint(prefix, region))
      expect(payload().model).toBe('MiniMax-M3.1-Flash')
      expect(payload().messages[0].content).toContain('Prior work')
    })

    it('preserves reasoning, tool results and prior media through tool turns and session persistence', async () => {
      vi.stubEnv('MINIMAX_REGION', region)
      const assistant = prefix === 'minimax'
        ? {
            role: 'assistant', content: null,
            reasoning_details: [{ type: 'reasoning.text', text: 'Inspect first' }],
            reasoning_content: 'Inspect first',
            tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'inspect', arguments: '{}' } }]
          }
        : {
            role: 'assistant',
            content: [
              { type: 'thinking', thinking: 'Inspect first', signature: 'fixture-signature' },
              { type: 'tool_use', id: 'call-1', name: 'inspect', input: {} }
            ]
          }
      const priorMessage = {
        role: 'user',
        content: prefix === 'minimax'
          ? [{ type: 'image_url', image_url: { url: 'data:image/png;base64,fixture' } }]
          : [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'fixture' } }]
      }
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => prefix === 'minimax'
            ? { choices: [{ message: assistant }] }
            : { content: assistant.content, stop_reason: 'tool_use' }
        })
        .mockResolvedValueOnce({ ok: true, json: async () => reply(prefix) })
      const onToolCall = vi.fn().mockResolvedValue('state inspected')
      const onSessionUpdate = vi.fn()
      const result = await executeAgent({
        model: prefix, apiKey: 'fixture-key', prompt: 'Continue', systemPrompt: 'Be helpful',
        toolDefs, onToolCall, onSessionUpdate, maxTurns: 2,
        previousMessages: [priorMessage] as unknown as AgentExecutionOptions['previousMessages']
      })
      expect(result.exitCode).toBe(0)
      expect(mockFetch).toHaveBeenCalledTimes(2)
      expect(onToolCall).toHaveBeenCalledWith('inspect', {})
      for (let call = 0; call < 2; call++) {
        expect(mockFetch.mock.calls[call][0]).toBe(endpoint(prefix, region))
        expect(payload(call).model).toBe('MiniMax-M3.1-Flash')
        expect(payload(call).messages).toContainEqual(priorMessage)
      }
      expect(payload(1).messages).toContainEqual(assistant)
      const toolResult = prefix === 'minimax'
        ? { role: 'tool', tool_call_id: 'call-1', content: 'state inspected' }
        : { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'state inspected' }] }
      expect(payload(1).messages).toContainEqual(toolResult)
      expect(onSessionUpdate).toHaveBeenCalledOnce()
      expect(onSessionUpdate.mock.calls[0][0]).toContainEqual(assistant)
      expect(onSessionUpdate.mock.calls[0][0]).toContainEqual(toolResult)
      expect(onSessionUpdate.mock.calls[0][0]).toContainEqual(priorMessage)
      expect(onSessionUpdate.mock.calls[0][0].some((message: { role: string }) => message.role === 'system')).toBe(false)
    })
  })
})
