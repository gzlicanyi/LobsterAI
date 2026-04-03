import { test, expect } from 'vitest';
import { IMStore } from './imStore';

class FakeDb {
  imConfig: Map<string, string>;

  constructor() {
    this.imConfig = new Map();
  }

  run(sql: string, params: unknown[] = []) {
    if (sql.includes('INSERT INTO im_config')) {
      this.imConfig.set(String(params[0]), String(params[1]));
      return;
    }

    if (sql.includes('INSERT OR REPLACE INTO im_config')) {
      this.imConfig.set(String(params[0]), String(params[1]));
      return;
    }

    if (sql.includes('DELETE FROM im_config WHERE key = ?')) {
      this.imConfig.delete(String(params[0]));
      return;
    }

    if (sql.includes('DELETE FROM im_config')) {
      this.imConfig.clear();
    }
  }

  exec(sql: string, params: unknown[] = []) {
    if (sql.includes('SELECT value FROM im_config WHERE key = ?')) {
      const value = this.imConfig.get(String(params[0]));
      return value === undefined ? [] : [{ values: [[value]] }];
    }
    return [];
  }
}

test('IMStore persists conversation reply routes by platform and conversation ID', () => {
  const db = new FakeDb();
  let saveCount = 0;
  const store = new IMStore(db as unknown as ConstructorParameters<typeof IMStore>[0], () => {
    saveCount += 1;
  });

  expect(store.getConversationReplyRoute('dingtalk', '__default__:conv-1')).toBe(null);

  store.setConversationReplyRoute('dingtalk', '__default__:conv-1', {
    channel: 'dingtalk-connector',
    to: 'group:cid-42',
    accountId: '__default__',
  });

  expect(store.getConversationReplyRoute('dingtalk', '__default__:conv-1')).toEqual({
    channel: 'dingtalk-connector',
    to: 'group:cid-42',
    accountId: '__default__',
  });
  expect(store.getConversationReplyRoute('telegram', '__default__:conv-1')).toBe(null);
  expect(saveCount >= 2).toBeTruthy();
});
