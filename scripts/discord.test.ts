import { test, expect } from "bun:test";
import { message } from "./discord";

const entry = {
  slug: "echowm-omnimodal-world-model",
  title: "EchoWM: Enterable Omnimodal World Models with Unified Camera Intent",
  subtitle: "A world model that jointly generates 720p video, sound, music, and speech.",
  tags: ["EchoWM", "World Models", "Omnimodal Generation", "6-DoF Trajectory", "Camera Intent", "Video Generation"],
};

test("links the entry and its social card", () => {
  const [embed] = message(entry).embeds;
  expect(embed.url).toBe("https://paperlens.io/idea/echowm-omnimodal-world-model/");
  expect(embed.image.url).toBe("https://paperlens.io/og/echowm-omnimodal-world-model.png");
  expect(embed.title).toBe(entry.title);
  expect(embed.description).toBe(entry.subtitle);
});

test("shows at most five tags", () => {
  expect(message(entry).embeds[0].footer?.text).toBe(
    "EchoWM · World Models · Omnimodal Generation · 6-DoF Trajectory · Camera Intent",
  );
  expect(message({ ...entry, tags: [] }).embeds[0].footer).toBeUndefined();
});

test("keeps a long title inside Discord's 256-character cap", () => {
  const title = message({ ...entry, title: "x".repeat(400) }).embeds[0].title;
  expect(title.length).toBe(256);
  expect(title.endsWith("…")).toBe(true);
});

test("a title cannot ping the channel", () => {
  expect(message({ ...entry, title: "@everyone look" }).allowed_mentions.parse).toEqual([]);
});
