# SplitSnap: design brief and system

Status: **built and tested in demo mode (2026-10-09)**; Font check done: Newsreader and Atkinson Hyperlegible Next ship the rupee sign; Spline Sans Mono does not (not used yet).

## 1. What this is, and the feeling

A phone tool used at a table, one-handed, in dim restaurant light, by a hungry person who wants an argument to *not* happen. Everything depends on two things: **numbers that are unmistakable**, and **a split that feels fair and explained**.

**Concept: the counter slip.** The interface is a well-set restaurant check: warm uncoated paper, near-black ink, hairline rules, dot leaders between a dish and its price, a red pen for corrections, a highlighter for "look at this". No cards, no chrome. The bill is the interface; the app stays out of its way.

It should feel: calm, exact, a little tactile, quietly Indian without costume (correct ₹ and lakh/crore grouping, GST vocabulary, no rangoli/tricolour/paisley clichés).

## 2. What a generic Claude would produce, and what we refuse

This is the list I would reach for by reflex. **Every item below is banned.**

### Fonts: banned
Inter, Roboto, Arial, Helvetica, Segoe UI / `system-ui` as the visible identity, SF Pro, Open Sans, Lato, Poppins, Montserrat, Nunito, Raleway, DM Sans, Plus Jakarta Sans, Outfit, Manrope, Sora, Urbanist, Space Grotesk (the "AI product" font), Geist, JetBrains Mono / Fira Code as decoration, Courier / Courier New as a "receipt" costume, Playfair Display + Inter pairing, Instrument Serif and Fraunces (now the default "tasteful" picks), Comic/handwriting script fonts for "friendly". No gradient text. No text-shadow. No letter-spaced ALL-CAPS eyebrow label above every heading. No font-weight 800/900 headlines.

### Colour: banned
Indigo/violet/blue primaries (`#6366F1`, `#8B5CF6`, `#3B82F6`, `#2563EB`), purple-to-pink or blue-to-cyan gradients, mesh gradients and gradient blobs, neon accents on near-black, "slate-900 + glowing" dark mode, pure `#FFFFFF` page background, cool grey `#F9FAFB`/`#F3F4F6` page background, pure `#000000` text, Material/Bootstrap status colours (`#22C55E` green, `#EF4444` red as fills), rainbow category colours, opacity-tinted coloured badges (blue text on blue-10% pill).

### Shape, boxes, depth: banned
Cards (rounded box + shadow + border) as the default container for every thing; `border-radius` 12-24px anywhere; pill buttons and pill chips; `box-shadow` stacks (`shadow-md/lg/xl`), soft glows, inner shadows; glassmorphism / `backdrop-filter: blur`; neumorphism; left-border accent bars on cards; bento grids of equal tiles; three-column "feature" rows; icon-in-a-tinted-circle; zebra-striped bordered tables; outlined inputs with floating labels; iOS-style segmented controls and toggle switches; bottom-sheet "grab handle" clones; skeleton shimmer; full-width gradient hero with a CTA; centred-everything layouts; sticky blurred header.

### Backgrounds and decoration: banned
Dot grids, noise/grain overlays used as decoration, wavy dividers, blurred colour orbs, stock illustrations, 3D blobs, emoji (🍕🍽️💸🎉) used as icons or bullets, confetti, torn/zig-zag "receipt edge" gimmicks, faux paper curl, rotated stickers everywhere.

### Copy and motion: banned
"Seamless", "effortless", "powerful", "magic", "AI-powered", "Let's get started", "Oops!", "Welcome back", exclamation marks, emoji, apologetic cuteness. No bouncy springs, parallax, auto-playing or looping animation, hover-only affordances (this is a phone), page transitions over 150 ms, animated number count-ups (numbers must never be in motion when read).

## 3. What we do instead

### 3.1 Palette (WCAG contrast measured, text pairs)

Light ("paper", default):

| Token | Hex | Use | Contrast on paper |
|---|---|---|---|
| `--paper` | `#F3EEE3` | page | |
| `--paper-2` | `#E8E1D2` | wells (recessed areas) | |
| `--ink` | `#1C1A16` | text, primary button | 15.0 |
| `--ink-2` | `#5B564A` | secondary text | 6.3 (5.6 on paper-2) |
| `--rule` | `#CFC6B2` | hairlines | non-text |
| `--vermilion` | `#B23A21` | red pen: errors, mismatches, destructive | 5.2 |
| `--tick` | `#2E5E3B` | reconciled / matches | 6.5 |
| `--marker` | `#F1D35A` at 60% | highlighter under low-confidence fields | ink on it 12.9 |

Dark ("chalkboard ink", follows `prefers-color-scheme`): bg `#15130F`, bg-2 `#1F1C16`, text `#ECE5D5` (14.8), text-2 `#A39C8B` (6.8), rule `#3A352B`, vermilion `#EE7B5F` (6.7), tick `#7FB38A` (7.7), marker `#D9B53C` at 35% (text on it 6.7).

Rules of use: colour carries meaning only. **Ink is the brand.** Vermilion appears only where something needs a human decision. Marker appears only on uncertainty. Nothing is decorative-coloured.

People marks (8 muted inks, always paired with an initial so colour is never the only signal): oxblood `#8A2D22`, indigo ink `#2F3E6B`, teal ink `#2A6168`, ochre `#8A6A12`, plum `#5E3A5C`, slate `#4A5560`, rust `#A4552B`, olive `#5B6322`.

### 3.2 Typography (self-hosted, SIL OFL, Latin subset)

| Role | Family | Why |
|---|---|---|
| Headings, big amounts, margin notes | **Newsreader** (variable, optical size) | A book/newspaper serif, warm and exact at large sizes; italic gives the "pen in the margin" voice |
| UI, labels, body, all numerals | **Atkinson Hyperlegible Next** | Designed by the Braille Institute so 1/l/I and 0/O never blur: right for prices read in dim light. Tabular figures on |
| Raw-bill strip only (what the model read) | **Spline Sans Mono** | A calm mono for the "as printed" view; not used elsewhere |
| ₹ fallback | an Indian-designed OFL face (Mukta or Hind), `unicode-range: U+20B9` | Many fonts lack ₹ or draw it badly; verify before ship |

Scale (ratio 1.25, base 16px; inputs never below 16px to stop iOS zoom): 13 / 16 / 20 / 25 / 31 / 39, plus **one display size (48px, Newsreader 500) used only for the final amount**. Line-height 1.5 body, 1.15 display. Sentence case everywhere. Tabular, lining figures for money. Weights used: 400, 500, 600 only.

Money format: `₹` prefix, Indian grouping (`₹1,23,456.00` via `Intl.NumberFormat('en-IN')`), true minus `−` (U+2212), discounts shown as `−₹50.00` in ink, never in red.

### 3.3 Shape and depth

- **Radius 0** for structure; **2px** for inputs and buttons. Fully round only for the 24px person mark.
- **No shadows at all.** Separation comes from hairlines (1px `--rule`), whitespace (40px between sections) and recessed `--paper-2` wells.
- One tactile exception: the primary button has a hard 2px solid under-edge (a stamped look) that closes on press.
- **Inputs are underlines**, like writing on a form: bottom border only; focus = 2px ink underline and a vermilion caret. No boxes.
- **Dot leaders** between item and amount (`.....`) are the signature texture. Rows are separated by nothing but space.
- **Margin notes** (Newsreader italic 15px, `--ink-2`) are how the app talks: "the items add up to ₹740.00, the bill says ₹741.50".

### 3.4 Layout

Single column, max 28rem (448px), 20px gutters, centred on the page on larger screens with flat paper either side (no device mockup, no side panels). Primary action pinned in a bottom bar: a hairline above it, the running total on the left in the display style, one button on the right. Targets at least 48px. One scrolling document per step; no nested scroll areas, no modals (use inline expansion).

### 3.5 Motion and feedback

120 ms ease-out on state changes only (underline thickening, row insert). The "reading" state is a single 1px ink rule sweeping across a blank ruled bill (static text under `prefers-reduced-motion`). Nothing bounces, nothing counts up. Haptics: none. Sound: none.

### 3.6 Icons

None by default; words are clearer. Exactly six hand-drawn 1.5px stroke SVGs (camera, plus, close, check, copy, info), `currentColor`, no icon font, no library.

### 3.7 Voice

Plain, terse, lowercase-leaning headings, no exclamation marks. Examples: "Add the bill", "Check what we read", "Who's eating", "Who had what", "The split". Errors say what happened and what to do: "That photo is too dark to read. Move to the light and try again." Empty state: a blank bill with dotted lines and one sentence.

## 4. Special treatments (the memorable parts)

| Situation | Treatment |
|---|---|
| Low model confidence (R2) | Highlighter band behind the value, as if someone marked it; tap to see the model's raw reading |
| Arithmetic mismatch (R3) | Red-pen double underline on the number plus a margin note in italic stating both sides |
| Reconciled total | Small green tick next to "Total", margin note "adds up" |
| Manual override (R7) | An "adjusted by hand" stamp (vermilion border and text, rotated −2°, the only rotated element) beside the amount, original value shown struck through |
| Charges (R5) | Own block under the items: Items / Taxes / Service charge / Packaging / Discount / Rounding / Total, each on a dot leader |
| Charge share (R5) | In each person's slip, each charge listed with its amount and the share it came from ("12.9% of your ₹260 subtotal") |
| Retake prompt (R2) | A plain sentence in a vermilion-ruled band: what is wrong, what to do; "use it anyway" stays available |
| Names (R8) | Typed freely; each gets a mark (colour + initial); remove with a clear confirm line if they have items |

## 5. Screens (five steps, one document each)

```
1  Photo                      2  Check the bill              3  Who's eating
-------------------          -------------------            -------------------
 SplitSnap                    Check what we read              Who's eating
                              Hotel Sagar        12/09/25     Riya   ........  x
 [ Take a photo  ]            ----------------------------    Aman   ........  x
 or choose a file             Paneer Butter  1  220.00        ____________ add
                              Garlic Naan    3  120.00  ░░    
 (empty ruled bill)           Cold Coffee    1  140.00        ----------------
                              ----------------------------    [ Next: who had what ]
 ----------------------       Items            740.00
  total   —     [ Read ]      CGST              18.50
                              Service charge    40.00
                              Discount         −50.00
                              Total            767.00 ✓
                              note: Naan price looks low
                              ----------------------------
                               ₹767.00      [ Add people ]

4  Who had what               5  The split
-------------------          -------------------
 Garlic Naan  120.00          ₹767.00
  Riya ×1  Aman ×1  Sara ×1   Riya ........ ₹293.50
 Paneer B.    220.00            Paneer (full) 220.00
  Riya ×1  Aman  Sara          Naan (1/3)     40.00
 [ everyone ] [ equally ]       Tax + service 33.50  (12.9% of ₹260)
                              Aman ........ ₹ …
 ₹767.00 left to assign 0      [ adjust an amount ]  [ copy summary ]
  [ See the split ]            note: how this was worked out (expand)
```

Step 2 doubles as the correction UI (R3): every cell is an underlined input; add and delete a row inline; each edit calls `/validate` and the margin notes and ticks update in place. Step 5 carries the explanation (R6) as expandable text per person and overrides (R7) as inline edits on the amount.

## 6. Build constraints

- Vanilla HTML, CSS and JS. No framework, no CDN, no analytics, no runtime network calls except our own API (open-source and privacy: the bill photo never leaves the machine running the model).
- Files: `splitsnap/static/{index.html, app.css, app.js, fonts/*.woff2}` served by the FastAPI app; the demo mode of the backend lets the whole flow run without a trained model.
- Fonts are subset to Latin plus `₹` and self-hosted; licences (OFL) listed in the README next to the models and libraries (R4 declaration).
- Tokens live as CSS custom properties on `:root`; dark values redefined under `prefers-color-scheme: dark`.
- No inline styles for tokens, no `!important`, no utility-class framework.

## 7. How I will check it (no browser extras needed)

- Headless Firefox screenshots at 360x740, 390x844, 768x1024, 1280x800, light and dark, for each of the five steps and for three states (clean bill, messy bill with flags, override applied).
- A script computes contrast for every text/background pair from the CSS tokens; fail below 4.5:1 (3:1 for large text).
- No horizontal scroll at 320px; every control at least 48px; full keyboard path; focus always visible; `aria-live` on totals.
- Acceptance bar: if the screenshot could be mistaken for a default component-library template, it is redone.

## 8. Open decisions (cheap to change now)

1. Final font check: confirm Newsreader / Atkinson Hyperlegible Next / Spline Sans Mono all ship ₹ or fall back cleanly; swap to Literata or Public Sans only if a glyph check fails.
2. Whether to ship a manual light/dark toggle (default: no, follow the system).
3. Whether the raw-bill strip (what the model read) is shown by default or behind "see what we read" (default: behind a link, to keep step 2 short).
