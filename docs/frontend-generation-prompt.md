# AstroStack Frontend Generation Prompt

## Project Overview

Generate a modern, minimalist web interface for **AstroStack**, an astrophotography processing pipeline. The interface must provide a premium user experience similar to high-end web design studios.

## Technical Stack Requirements

- **Framework**: React with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with custom design tokens
- **State Management**: React Query for server state, Zustand for UI state
- **HTTP Client**: Axios with streaming support
- **Real-time**: WebSocket client for live updates
- **UI Components**: Radix UI (headless) + custom styling

## API Endpoints (Base URL: `/api/v1`)

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/sessions` | List sessions (paginated) |
| GET | `/sessions/{session_id}` | Get session details |
| POST | `/sessions/{session_id}/process?preset={preset}&profile_id={id}` | Start processing |
| POST | `/sessions/{session_id}/cancel` | Cancel active job |

**Query Parameters:**
- `preset`: `quick` | `standard` | `quality` | `advanced` (default: `standard`)
- `profile_id`: UUID (required only for `advanced` preset)

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/jobs/{job_id}` | Get job status + step results |
| GET | `/jobs/{job_id}/output/preview` | Download JPEG preview |
| GET | `/jobs/{job_id}/output/fits` | Download final FITS file |

### Profiles (Advanced Mode)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/profiles` | List all profiles |
| POST | `/profiles` | Create new profile |
| GET | `/profiles/{profile_id}` | Get profile details |
| PUT | `/profiles/{profile_id}` | Update profile |
| DELETE | `/profiles/{profile_id}` | Delete profile |

## WebSocket Events (Endpoint: `/api/v1/ws`)

Connect with query parameter: `?session_id={id}`

### Incoming Events

```typescript
// Job status updates
{ "type": "job_status", "job_id": "uuid", "status": "pending" | "running" | "completed" | "failed" | "cancelled", "current_step": "string | null" }

// Step progress
{ "type": "step_update", "step_name": "string", "step_index": number, "status": "pending" | "running" | "success" | "failed" | "skipped" | "retrying", "attempt_count": number, "error_code": "string | null" }

// Session updates
{ "type": "session_status", "status": "pending" | "ready" | "processing" | "completed" | "failed" | "cancelled" }

// Error events
{ "type": "error", "code": "string", "message": "string" }
```

## Data Models

### SessionStatus
```typescript
type SessionStatus = "pending" | "ready" | "processing" | "completed" | "failed" | "cancelled";
```

### JobStatus
```typescript
type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled" | "paused";
```

### StepStatus
```typescript
type StepStatus = "pending" | "running" | "success" | "failed" | "skipped" | "retrying";
```

### ProfilePreset
```typescript
type ProfilePreset = "quick" | "standard" | "quality" | "advanced";
```

### SessionRead
```typescript
interface SessionRead {
  id: string;           // UUID
  name: string;
  inbox_path: string;
  status: SessionStatus;
  input_format: "fits" | "raw_dslr" | "mixed" | null;
  frame_count_lights: number;
  frame_count_darks: number;
  frame_count_flats: number;
  frame_count_bias: number;
  object_name: string | null;
  ra: number | null;
  dec: number | null;
  created_at: string;   // ISO 8601
  updated_at: string;   // ISO 8601
}
```

### JobRead
```typescript
interface JobRead {
  id: string;
  session_id: string;
  profile_preset: ProfilePreset;
  status: JobStatus;
  current_step: string | null;
  started_at: string | null;
  completed_at: string | null;
  error_code: string | null;
  output_fits_path: string | null;
  output_tiff_path: string | null;
  output_preview_path: string | null;
  created_at: string;
  steps: JobStepRead[];
}

interface JobStepRead {
  id: string;
  step_name: string;
  step_index: number;
  status: StepStatus;
  attempt_count: number;
  started_at: string | null;
  completed_at: string | null;
  error_code: string | null;
  output_metadata: Record<string, unknown> | null;
}
```

### ProcessingProfileConfig (Pipeline Steps)
```typescript
interface ProcessingProfileConfig {
  // Each step can have custom parameters
  preprocessing?: {
    enable?: boolean;
    hot_pixel_threshold?: number;
    cosmic_ray_rejection?: boolean;
  };
  raw_conversion?: {
    enable?: boolean;
    debayer?: boolean;
    white_balance?: "auto" | "manual";
  };
  star_separation?: {
    enable?: boolean;
    sensitivity?: number;
    stars_threshold?: number;
  };
  gradient_removal?: {
    enable?: boolean;
    algorithm?: "automatic" | "manual";
    degree?: number;
  };
  denoise?: {
    enable?: boolean;
    method?: "ai" | "traditional";
    strength?: number;
    model?: string;
  };
  plate_solving?: {
    enable?: boolean;
    solver?: "astap" | "astrometry";
    timeout?: number;
  };
  stretch_color?: {
    enable?: boolean;
    algorithm?: "arcsinh" | "log" | "linear";
    factor?: number;
  };
  super_resolution?: {
    enable?: boolean;
    scale?: number;
    model?: string;
  };
  sharpen?: {
    enable?: boolean;
    method?: " Richardson-Lucy" | "deconvolution";
    iterations?: number;
  };
  export?: {
    enable?: boolean;
    format?: "fits" | "tiff" | "jpeg";
    quality?: number;
  };
}
```

## File Upload Requirements

- Use **chunked upload** for large files (FITS files can be hundreds of MB)
- Implement resumable uploads with progress tracking
- Endpoint: POST `/sessions/upload` with `Upload-Start`, `Upload-Chunk`, `Upload-Finalize` headers
- Alternative: Stream upload with `Content-Range` header

## UI/UX Requirements

### Design System

1. **Color Palette** (Dark Theme - Space-inspired):
   - Background: `#0a0a0f` (deep space black)
   - Surface: `#12121a` (card backgrounds)
   - Surface Elevated: `#1a1a24` (modals, dropdowns)
   - Border: `#2a2a3a` (subtle borders)
   - Primary: `#6366f1` (indigo - main actions)
   - Primary Hover: `#818cf8`
   - Accent: `#22d3ee` (cyan - highlights)
   - Success: `#10b981` (green)
   - Warning: `#f59e0b` (amber)
   - Error: `#ef4444` (red)
   - Text Primary: `#f8fafc`
   - Text Secondary: `#94a3b8`
   - Text Muted: `#64748b`

2. **Typography**:
   - Font Family: "Inter" for UI, "JetBrains Mono" for data/code
   - Headings: 600 weight
   - Body: 400 weight
   - Font Sizes: 12px (small), 14px (body), 16px (large), 20px (h3), 24px (h2), 32px (h1)

3. **Spacing**: 4px base unit (4, 8, 12, 16, 24, 32, 48, 64)

4. **Border Radius**: 6px (buttons), 8px (cards), 12px (modals), 9999px (pills)

5. **Animations**:
   - Duration: 150ms (micro), 300ms (standard), 500ms (emphasis)
   - Easing: `cubic-bezier(0.4, 0, 0.2, 1)`
   - Subtle fade/slide transitions
   - Pulsing glow for active processing

### Layout Structure

1. **Header**: Logo left, navigation center, profile/settings right
2. **Main Content**: 
   - **Dashboard View**: Grid of session cards with status indicators
   - **Session Detail View**: Split view - file list left, processing panel right
   - **Profile Editor**: Form-based with sections for each pipeline step

### User Modes

#### Simple Mode (Presets)
- Single page with 3 preset buttons: Quick, Standard, Quality
- Drag-and-drop file upload zone
- One-click processing start
- Live progress indicator (step by step)
- Download buttons on completion

#### Advanced Mode (Custom Profiles)
- Toggle between Simple/Advanced modes
- Full profile management (CRUD)
- Per-step configuration toggles and parameters
- Profile save/load functionality
- Expert-level controls (advanced settings)

### Key UI Components

1. **Session Card**: Name, date, frame counts, status badge, action menu
2. **Upload Zone**: Large drag-drop area, chunked progress bar, file list
3. **Progress Panel**: Step-by-step progress, current step highlight, elapsed time
4. **Preset Selector**: Large tappable buttons with descriptions
5. **Profile Form**: Collapsible sections, toggle switches, inline validation
6. **Status Badge**: Color-coded pill with icon
7. **Progress Stepper**: Horizontal or vertical step indicators

### Responsive Breakpoints
- Mobile: < 640px (single column, stacked)
- Tablet: 640px - 1024px (two columns)
- Desktop: > 1024px (full layout)

## Code Requirements

1. Use TypeScript with strict mode
2. All text, comments, and documentation in English
3. Implement proper error handling with user-friendly messages
4. Use React Query for caching and optimistic updates
5. Implement proper loading states and skeleton screens
6. Accessibility: ARIA labels, keyboard navigation, focus management

## Deliverables

1. React application with all components
2. API service layer with TypeScript types
3. WebSocket hook for real-time updates
4. Form validation schemas
5. Responsive layouts for all breakpoints
6. Loading and error states
