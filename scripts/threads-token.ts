#!/usr/bin/env bun
/**
 * Threads access-token helper.
 *
 * Threads hands you a short-lived token (1 hour). You exchange it for a
 * long-lived one (60 days), and refresh that before it lapses. A long-lived
 * token that goes 60 days without a refresh is dead and cannot be revived —
 * you have to re-authorize from scratch. This wraps the three calls.
 *
 *   bun scripts/threads-token.ts check     --token <long-lived>
 *   bun scripts/threads-token.ts exchange  --token <short-lived> --secret <app-secret>
 *   bun scripts/threads-token.ts refresh   --token <long-lived>
 *   bun scripts/threads-token.ts rotate    --token <long-lived>   # refresh + store
 *
 * `--token` falls back to THREADS_ACCESS_TOKEN, `--secret` to THREADS_APP_SECRET.
 */

const GRAPH = "https://graph.threads.net";

function arg(name: string): string | undefined {
	const i = Bun.argv.indexOf(`--${name}`);
	return i === -1 ? undefined : Bun.argv[i + 1];
}

function days(seconds: number): string {
	return `${Math.floor(seconds / 86400)} days`;
}

async function get(path: string, params: Record<string, string>): Promise<any> {
	const res = await fetch(`${GRAPH}${path}?${new URLSearchParams(params)}`);
	const body = await res.json().catch(() => ({}));
	if (!res.ok) {
		const err = (body as any)?.error;
		throw new Error(`Threads API ${res.status} on ${path}: ${err?.message ?? JSON.stringify(body)}`);
	}
	return body;
}

const command = Bun.argv[2];
const token = arg("token") ?? process.env.THREADS_ACCESS_TOKEN;
const secret = arg("secret") ?? process.env.THREADS_APP_SECRET;

if (!command || !["check", "exchange", "refresh", "rotate"].includes(command)) {
	console.error("usage: bun scripts/threads-token.ts <check|exchange|refresh|rotate> [--token …] [--secret …]");
	process.exit(2);
}
if (!token) {
	console.error("No token. Pass --token or set THREADS_ACCESS_TOKEN.");
	process.exit(2);
}

try {
	if (command === "check") {
		const me = await get("/v1.0/me", { fields: "id,username,threads_profile_picture_url", access_token: token });
		console.log(`token works — @${me.username} (id ${me.id})`);
		console.log("\nSet these on the repository:");
		console.log(`  gh secret set THREADS_ACCESS_TOKEN --body '<this token>'`);
		console.log(`  gh variable set THREADS_USER_ID --body '${me.id}'`);
	}

	if (command === "exchange") {
		if (!secret) {
			console.error("Exchange needs the app secret. Pass --secret or set THREADS_APP_SECRET.");
			process.exit(2);
		}
		const out = await get("/access_token", {
			grant_type: "th_exchange_token",
			client_secret: secret,
			access_token: token,
		});
		console.log(`long-lived token (valid ${days(out.expires_in ?? 0)}):\n`);
		console.log(out.access_token);
		console.log(`\n  gh secret set THREADS_ACCESS_TOKEN --body '${out.access_token}'`);
	}

	if (command === "refresh") {
		const out = await get("/refresh_access_token", {
			grant_type: "th_refresh_token",
			access_token: token,
		});
		console.log(`refreshed (valid another ${days(out.expires_in ?? 0)}):\n`);
		console.log(out.access_token);
		console.log(`\n  gh secret set THREADS_ACCESS_TOKEN --body '${out.access_token}'`);
	}
	if (command === "rotate") {
		// Unattended renewal. The new token must never reach the run log, so it
		// is masked before it is used and written straight into the secret.
		const out = await get("/refresh_access_token", {
			grant_type: "th_refresh_token",
			access_token: token,
		});
		const next = out.access_token as string;
		if (!next) throw new Error(`No access_token in the refresh response: ${JSON.stringify(out)}`);

		if (process.env.GITHUB_ACTIONS === "true") {
			// Registers the value with the runner so it is redacted everywhere.
			console.log(`::add-mask::${next}`);
		}

		const validFor = out.expires_in ?? 0;
		if (next === token) {
			console.error("The API returned the same token; nothing to store.");
		} else {
			const proc = Bun.spawnSync(["gh", "secret", "set", "THREADS_ACCESS_TOKEN"], {
				stdin: new TextEncoder().encode(next),
				stdout: "pipe",
				stderr: "pipe",
			});
			if (proc.exitCode !== 0) {
				throw new Error(
					`Refreshed the token but could not store it: ${proc.stderr.toString().trim()}\n` +
						"The PAT in GH_TOKEN needs the repository Secrets permission set to read and write.",
				);
			}
			console.error("stored the refreshed token in THREADS_ACCESS_TOKEN");
		}

		console.error(`valid for another ${days(validFor)}`);
		if (process.env.GITHUB_OUTPUT) {
			// Must append: this file collects the outputs of every step.
			const { appendFile } = await import("node:fs/promises");
			await appendFile(process.env.GITHUB_OUTPUT, `valid_days=${Math.floor(validFor / 86400)}\n`);
		}
	}
} catch (err) {
	console.error(err instanceof Error ? err.message : String(err));
	process.exit(1);
}

// Top-level await requires this file to be a module.
export {};
