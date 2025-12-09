# Frontend Compatibility Checklist

Quick pass to verify the Lexi web frontend after layout or build changes.

- [ ] `npm run build` from `frontend` succeeds.
- [ ] Desktop Chrome: load app, send/receive chat, scroll history, avatar tools open.
- [ ] Desktop Safari: load app, chat flows, scrollable areas work, tour/legal modals render.
- [ ] Desktop Firefox/Edge: chat + downloads/buttons behave, no layout shifts on resize.
- [ ] iOS Safari: chat input stays visible with keyboard open; send works; no stuck scrollbars.
- [ ] Android Chrome: no horizontal scroll; chat input and avatar interactions feel responsive.
