/**
 * IOFT Humanoid Robot AI Cockpit
 * Ultra Low-Latency STT, TTS, Locomotion & Telemetry Controller
 */

// ============================================================================
// State Management
// ============================================================================
const state = {
  isListening: false,
  isProcessing: false,
  isSpeaking: false,
  continuousMode: false, // Default to click-to-speak for highest reliability
  currentAuthority: "AI",
  robotOnline: false,
  speedProfile: "slow",
  speedPwm: 70,
  steps: 1,
  ttsEnabled: true,
  ttsVoice: null,
  ttsRate: 1.0,
  availableVoices: [],
};

// ============================================================================
// Web Audio Synthesizer & Microphone Analyzer
// ============================================================================
class AudioController {
  constructor() {
    this.ctx = null;
    this.analyser = null;
    this.micStream = null;
    this.animId = null;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
  }

  playBeep(freq, type, duration, gainValue = 0.1) {
    try {
      this.init();
      if (!this.ctx) return;

      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);

      gain.gain.setValueAtTime(gainValue, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {
      console.warn("Audio cue error:", e);
    }
  }

  startListeningTone() {
    this.playBeep(587.33, "sine", 0.12, 0.08); // D5
    setTimeout(() => this.playBeep(880, "sine", 0.15, 0.08), 80); // A5
  }

  successChime() {
    this.playBeep(523.25, "triangle", 0.1, 0.1); // C5
    setTimeout(() => this.playBeep(659.25, "triangle", 0.1, 0.1), 90); // E5
    setTimeout(() => this.playBeep(783.99, "triangle", 0.2, 0.1), 180); // G5
  }

  estopTone() {
    this.playBeep(880, "sawtooth", 0.15, 0.2);
    setTimeout(() => this.playBeep(440, "sawtooth", 0.25, 0.2), 160);
  }

  // Connect microphone stream and calculate real RMS decibel volume
  async startMicVisualizer(deviceId = null) {
    try {
      this.init();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

      if (this.micStream) {
        this.micStream.getTracks().forEach(t => t.stop());
        this.micStream = null;
      }

      const constraints = {
        audio: {
          echoCancellation: true,
          noiseSuppression: false, // Don't filter out normal voice
          autoGainControl: true,
        }
      };

      if (deviceId) {
        constraints.audio.deviceId = { exact: deviceId };
      }

      this.micStream = await navigator.mediaDevices.getUserMedia(constraints);
      await this.ctx.resume();

      const source = this.ctx.createMediaStreamSource(this.micStream);
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.3;
      source.connect(this.analyser);

      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const updateBars = () => {
        if (!this.micStream) return;

        this.analyser.getByteTimeDomainData(dataArray);
        
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          const val = (dataArray[i] - 128) / 128;
          sum += val * val;
        }
        const rms = Math.sqrt(sum / bufferLength);
        // Amplified RMS calculation: normal speaking reaches 30% - 90%
        const percent = Math.min(100, Math.round(rms * 450));

        const vuFill = document.getElementById("vu-meter-fill");
        if (vuFill) {
          vuFill.style.width = `${percent}%`;
        }

        this.animId = requestAnimationFrame(updateBars);
      };

      if (this.animId) cancelAnimationFrame(this.animId);
      this.animId = requestAnimationFrame(updateBars);

      logEvent("STT", "Microphone stream connected. Audio input active.");
      return true;
    } catch (e) {
      console.warn("Mic visualizer error:", e);
      logEvent("ERR", `Microphone access error: ${e.message}`);
      return false;
    }
  }

  stopMicVisualizer() {
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
    const vuFill = document.getElementById("vu-meter-fill");
    if (vuFill) {
      vuFill.style.width = "0%";
    }
  }

  async listAudioDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(d => d.kind === "audioinput");
      const select = document.getElementById("mic-device-select");
      if (!select) return;

      select.innerHTML = "";
      if (audioInputs.length === 0) {
        select.innerHTML = '<option value="">Default Microphone</option>';
        return;
      }

      audioInputs.forEach((device, index) => {
        const opt = document.createElement("option");
        opt.value = device.deviceId;
        opt.textContent = device.label || `Microphone ${index + 1}`;
        select.appendChild(opt);
      });
    } catch (e) {
      console.warn("enumerateDevices error:", e);
    }
  }
}

const audioCtrl = new AudioController();

// ============================================================================
// Text-to-Speech (TTS) Engine
// ============================================================================
class TTSEngine {
  constructor() {
    this.synth = window.speechSynthesis;
    this.initVoices();
    if (this.synth && speechSynthesis.onvoiceschanged !== undefined) {
      speechSynthesis.onvoiceschanged = () => this.initVoices();
    }
  }

  initVoices() {
    if (!this.synth) return;
    state.availableVoices = this.synth.getVoices();
    const select = document.getElementById("voice-select");
    if (!select) return;

    select.innerHTML = "";
    state.availableVoices.forEach((voice, i) => {
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = `${voice.name} (${voice.lang})`;
      
      // Auto-select preferred English voices
      if (
        !state.ttsVoice &&
        (voice.name.includes("Natural") ||
         voice.name.includes("Christopher") ||
         voice.name.includes("Jenny") ||
         voice.name.includes("David") ||
         voice.name.includes("Zira") ||
         (voice.lang.startsWith("en") && voice.default))
      ) {
        state.ttsVoice = voice;
        opt.selected = true;
      }
      select.appendChild(opt);
    });

    if (!state.ttsVoice && state.availableVoices.length > 0) {
      state.ttsVoice = state.availableVoices[0];
    }
  }

  speak(text, onComplete = null) {
    if (!state.ttsEnabled || !this.synth || !text) {
      if (onComplete) onComplete();
      return;
    }

    try {
      if (this.synth.speaking || this.synth.pending) {
        this.synth.cancel();
      }
      this.synth.resume();

      // Clean technical text for natural speech
      let clean = text
        .replace(/Controller acknowledged command\(s\); physical motion unverified - no sensors/gi, "")
        .replace(/\(Controller acknowledged.*?\)/gi, "")
        .replace(/\|/g, " ")
        .trim();

      if (!clean) {
        if (onComplete) onComplete();
        return;
      }

      const utterance = new SpeechSynthesisUtterance(clean);
      if (state.ttsVoice) utterance.voice = state.ttsVoice;
      utterance.rate = state.ttsRate;
      utterance.pitch = 1.0;

      utterance.onstart = () => {
        state.isSpeaking = true;
        updateOrbState();
        logEvent("TTS", `Speaking: "${clean}"`);
      };

      utterance.onend = () => {
        state.isSpeaking = false;
        updateOrbState();
        if (onComplete) onComplete();
      };

      utterance.onerror = (e) => {
        console.warn("TTS speak error:", e);
        state.isSpeaking = false;
        updateOrbState();
        if (onComplete) onComplete();
      };

      // Slight timeout prevents Chrome cancellation glitch
      setTimeout(() => {
        this.synth.speak(utterance);
      }, 50);

    } catch (err) {
      console.warn("TTS invocation error:", err);
      state.isSpeaking = false;
      updateOrbState();
      if (onComplete) onComplete();
    }
  }

  stop() {
    if (this.synth) this.synth.cancel();
    state.isSpeaking = false;
    updateOrbState();
  }
}

const tts = new TTSEngine();

// ============================================================================
// Direct 16kHz PCM WAV Audio Recorder (Dedicated Server STT Pipeline)
// ============================================================================
class AudioRecorder {
  constructor(audioContext) {
    this.ctx = audioContext;
    this.node = null;
    this.chunks = [];
    this.isRecording = false;
  }

  start(mediaStream) {
    this.chunks = [];
    this.isRecording = true;
    const source = this.ctx.createMediaStreamSource(mediaStream);
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    this.node.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const input = e.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(input));
    };
    source.connect(this.node);
    this.node.connect(this.ctx.destination);
  }

  stop() {
    this.isRecording = false;
    if (this.node) {
      try {
        this.node.disconnect();
      } catch (e) {}
      this.node = null;
    }

    let length = 0;
    for (const c of this.chunks) length += c.length;
    const fullBuffer = new Float32Array(length);
    let offset = 0;
    for (const c of this.chunks) {
      fullBuffer.set(c, offset);
      offset += c.length;
    }

    const wavBlob = this.encodeWAV(fullBuffer, this.ctx.sampleRate);
    this.chunks = [];
    return wavBlob;
  }

  encodeWAV(samples, inputSampleRate) {
    const targetSampleRate = 16000;
    const downsampled = this.downsampleBuffer(samples, inputSampleRate, targetSampleRate);
    const buffer = new ArrayBuffer(44 + downsampled.length * 2);
    const view = new DataView(buffer);

    const writeString = (v, off, string) => {
      for (let i = 0; i < string.length; i++) {
        v.setUint8(off + i, string.charCodeAt(i));
      }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + downsampled.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, targetSampleRate, true);
    view.setUint32(28, targetSampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); // 16-bit
    writeString(view, 36, 'data');
    view.setUint32(40, downsampled.length * 2, true);

    let idx = 44;
    for (let i = 0; i < downsampled.length; i++, idx += 2) {
      const s = Math.max(-1, Math.min(1, downsampled[i]));
      view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  downsampleBuffer(buffer, sampleRate, outSampleRate) {
    if (outSampleRate >= sampleRate) return buffer;
    const sampleRateRatio = sampleRate / outSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0, count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count > 0 ? (accum / count) : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }
}

// ============================================================================
// Dedicated Server-Side Speech-to-Text Pipeline (Python STT Engine)
// ============================================================================
class ServerSTTManager {
  constructor() {
    this.recorder = null;
    this.isRecording = false;
    this.recordStartTime = 0;
    this.timerInterval = null;
  }

  async toggle() {
    if (this.isRecording) {
      await this.stopAndTranscribe();
    } else {
      await this.startRecording();
    }
  }

  async startRecording() {
    audioCtrl.init();
    await audioCtrl.ctx.resume();

    const deviceSelect = document.getElementById("mic-device-select");
    const deviceId = deviceSelect ? deviceSelect.value : null;
    const connected = await audioCtrl.startMicVisualizer(deviceId);
    if (!connected || !audioCtrl.micStream) {
      logEvent("ERR", "Could not start microphone. Please check your mic connection.");
      return;
    }

    this.recorder = new AudioRecorder(audioCtrl.ctx);
    this.recorder.start(audioCtrl.micStream);
    this.isRecording = true;
    state.isListening = true;
    audioCtrl.startListeningTone();
    this.recordStartTime = Date.now();

    const btn = document.getElementById("voice-orb");
    const statusText = document.getElementById("orb-status");
    const pill = document.getElementById("stt-status-pill");

    if (btn) btn.classList.add("listening");
    if (pill) { pill.textContent = "RECORDING"; pill.style.color = "#ef4444"; }

    if (statusText) {
      statusText.textContent = "RECORDING... CLICK TO SEND";
      this.timerInterval = setInterval(() => {
        if (!this.isRecording) {
          clearInterval(this.timerInterval);
          return;
        }
        const secs = ((Date.now() - this.recordStartTime) / 1000).toFixed(1);
        statusText.textContent = `REC: ${secs}s (CLICK TO SEND)`;
      }, 200);
    }

    const transcriptEl = document.getElementById("transcript-text");
    if (transcriptEl) {
      transcriptEl.classList.remove("waiting");
      transcriptEl.innerHTML = `<span style="color: #ef4444; font-weight: 600;">[Recording audio from microphone... Speak now!]</span>`;
    }

    logEvent("STT", "Recording audio... Click button again when finished speaking.");
  }

  async stopAndTranscribe() {
    if (!this.isRecording || !this.recorder) return;
    this.isRecording = false;
    state.isListening = false;
    if (this.timerInterval) clearInterval(this.timerInterval);

    const btn = document.getElementById("voice-orb");
    const statusText = document.getElementById("orb-status");
    const pill = document.getElementById("stt-status-pill");

    if (btn) btn.classList.remove("listening");
    if (statusText) statusText.textContent = "TRANSCRIBING...";
    if (pill) { pill.textContent = "PROCESSING"; pill.style.color = "#38bdf8"; }

    const wavBlob = this.recorder.stop();
    const duration = ((Date.now() - this.recordStartTime) / 1000).toFixed(1);

    logEvent("STT", `Audio recorded (${duration}s, ${(wavBlob.size / 1024).toFixed(1)} KB). Sending to Python STT library...`);

    const transcriptEl = document.getElementById("transcript-text");
    if (transcriptEl) {
      transcriptEl.innerHTML = `<span style="color: #38bdf8;">[Transcribing audio with Python SpeechRecognition...]</span>`;
    }

    try {
      const langSelect = document.getElementById("stt-lang-select");
      const lang = langSelect ? langSelect.value : "en-IN";

      const formData = new FormData();
      formData.append("file", wavBlob, "voice.wav");

      const startTime = performance.now();
      const response = await fetch(`/api/v1/voice/stt?language=${encodeURIComponent(lang)}`, {
        method: "POST",
        body: formData,
      });

      const elapsed = Math.round(performance.now() - startTime);
      const data = await response.json();

      if (response.ok && data.success && data.transcript) {
        logEvent("STT", `Transcribed in ${elapsed}ms: "${data.transcript}"`);
        
        if (transcriptEl) {
          transcriptEl.innerHTML = `<strong style="color: #38bdf8; font-size: 0.95rem;">&gt; "${escapeHtml(data.transcript)}"</strong>`;
        }

        if (statusText) statusText.textContent = "MIC: PUSH TO TALK";
        if (pill) { pill.textContent = "READY"; pill.style.color = "#10b981"; }

        // Send to assistant locomotion or Q&A pipeline
        processVoiceCommand(data.transcript);

      } else {
        const msg = data.transcript || "No clear speech detected in recorded audio.";
        logEvent("STT", `STT: ${msg}`);
        if (transcriptEl) {
          transcriptEl.innerHTML = `<span style="color: #f59e0b;">[No intelligible speech heard. Speak clearly and try again.]</span>`;
        }
        if (statusText) statusText.textContent = "MIC: PUSH TO TALK";
        if (pill) { pill.textContent = "READY"; pill.style.color = "#10b981"; }
      }

    } catch (err) {
      logEvent("ERR", `Server STT error: ${err.message}`);
      if (statusText) statusText.textContent = "MIC: PUSH TO TALK";
      if (pill) { pill.textContent = "READY"; pill.style.color = "#10b981"; }
    }
  }
}

const serverSTT = new ServerSTTManager();
const stt = serverSTT; // Route UI calls to the dedicated server STT

// ============================================================================
// Intelligent Command Processing & Execution
// ============================================================================

function checkAuthorityCommand(text) {
  const lower = text.toLowerCase();
  
  if (lower.includes("emergency stop") || lower.includes("e stop") || lower.includes("halt robot")) {
    return "EMERGENCY_STOP";
  }
  if (lower.includes("manual") || lower.includes("teleop") || lower.includes("manual control") || lower.includes("set authority to manual")) {
    return "MANUAL";
  }
  if (lower.includes("ai control") || lower.includes("set authority to ai") || lower.includes("autonomous mode") || lower.includes("ai authority")) {
    return "AI";
  }
  return null;
}

function isQuestionOrChat(text) {
  const lower = text.toLowerCase();
  const questionWords = ["what", "who", "why", "how", "when", "where", "explain", "tell me", "can you", "hello", "hi "];
  
  const isQuestion = questionWords.some(w => lower.startsWith(w) || lower.includes(" " + w));
  const hasLocomotion = ["move", "forward", "backward", "back", "left", "right", "turn", "pivot", "drive", "reverse"].some(w => lower.includes(w));

  return isQuestion && !hasLocomotion;
}

async function processVoiceCommand(text) {
  state.isProcessing = true;
  updateOrbState();

  // Display user prompt in transcript
  const transcriptEl = document.getElementById("transcript-text");
  if (transcriptEl) {
    transcriptEl.innerHTML = `<strong style="color: #ffffff;">"${escapeHtml(text)}"</strong>`;
  }

  try {
    // 1. Authority change command
    const targetAuth = checkAuthorityCommand(text);
    if (targetAuth) {
      if (targetAuth === "EMERGENCY_STOP") {
        await handleEmergencyStop();
      } else {
        await setAuthority(targetAuth, true);
      }
      return;
    }

    // 2. Question / Conversational Q&A
    if (isQuestionOrChat(text)) {
      await handleConversation(text);
      return;
    }

    // 3. Locomotion Command via AI Assistant
    await handleLocomotionAssistant(text);

  } catch (err) {
    logEvent("ERR", `Execution error: ${err.message}`);
    tts.speak("Sorry, I encountered an error processing that command.");
  } finally {
    state.isProcessing = false;
    updateOrbState();

    if (state.continuousMode) {
      setTimeout(() => stt.start(), 1000);
    }
  }
}

async function handleLocomotionAssistant(promptText) {
  // If robot is currently offline, inform user
  if (!state.robotOnline) {
    logEvent("EXEC", `Sending command to AI Assistant (ESP32 currently offline)...`);
  } else {
    logEvent("EXEC", `Sending to AI Assistant: "${promptText}"`);
  }

  const startTime = performance.now();
  const response = await fetch("/api/v1/assistant", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: promptText })
  });

  const latency = Math.round(performance.now() - startTime);
  const data = await response.json();

  if (response.ok && data.executed) {
    audioCtrl.successChime();
    logEvent("EXEC", `Plan Executed (${latency}ms): ${data.plan?.plan_explanation || data.response_text}`);
    
    const speechText = data.plan?.plan_explanation 
      ? `Executing: ${data.plan.plan_explanation}` 
      : data.response_text;
    tts.speak(speechText);
  } else if (response.ok && !data.executed) {
    logEvent("EXEC", `AI Assistant: ${data.response_text}`);
    tts.speak(data.response_text);
  } else {
    const errorMsg = data.detail || "Command execution rejected.";
    logEvent("ERR", `Execution error: ${errorMsg}`);
    tts.speak(errorMsg);
  }
}

async function handleConversation(promptText) {
  logEvent("EXEC", `Asking AI Q&A: "${promptText}"`);

  const startTime = performance.now();
  const response = await fetch("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: promptText })
  });

  const latency = Math.round(performance.now() - startTime);
  if (response.ok) {
    const data = await response.json();
    audioCtrl.successChime();
    logEvent("EXEC", `AI Answer (${latency}ms): ${data.response}`);
    tts.speak(data.response);
  } else {
    const err = await response.json().catch(() => ({}));
    logEvent("ERR", `Chat error: ${err.detail || "Server error"}`);
    tts.speak("I am having trouble retrieving that answer right now.");
  }
}

async function setAuthority(targetAuthority, announceTTS = true) {
  // Target authority must be "AI" or "MANUAL"
  const auth = targetAuthority.toUpperCase() === "AI" ? "AI" : "MANUAL";

  try {
    const response = await fetch("/api/v1/commands/ownership", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ authority: auth })
    });

    if (response.ok) {
      const data = await response.json();
      state.currentAuthority = auth;
      updateAuthorityUI();
      audioCtrl.successChime();
      logEvent("EXEC", `Authority successfully set to: ${auth}`);

      if (announceTTS) {
        tts.speak(`Control authority set to ${auth === "AI" ? "AI Assistant" : "Manual Mode"}.`);
      }
    } else {
      const err = await response.json().catch(() => ({}));
      const detail = err.detail || "";
      
      // If switching AI -> MANUAL failed because ESP32 is offline (no STOP ack)
      if (response.status === 409 || detail.includes("STOP")) {
        logEvent("ERR", `Cannot transition to MANUAL while ESP32 is offline (requires confirmed STOP handshake). Authority remains AI.`);
        if (announceTTS) {
          tts.speak("Cannot switch to manual mode while the robot is offline.");
        }
      } else {
        logEvent("ERR", `Failed to set authority: ${detail || response.statusText}`);
      }
    }
  } catch (e) {
    logEvent("ERR", `Authority request error: ${e.message}`);
  }
}

async function handleEmergencyStop() {
  audioCtrl.estopTone();
  logEvent("EXEC", "EMERGENCY STOP TRIGGERED: Halting all motors");

  // Call assistant stop
  fetch("/api/v1/assistant", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "stop" })
  }).catch(() => {});

  tts.speak("Emergency stop activated. Halting all motors.");
}

// ============================================================================
// UI Updates & Visual Animations
// ============================================================================

function updateOrbState() {
  const btn = document.getElementById("voice-orb");
  const orbStatus = document.getElementById("orb-status");
  const pill = document.getElementById("stt-status-pill");
  if (!btn) return;

  btn.classList.remove("listening");

  if (state.isSpeaking) {
    if (orbStatus) orbStatus.textContent = "TTS: SPEAKING...";
    if (pill) { pill.textContent = "SPEAKING"; pill.style.color = "#a855f7"; }
  } else if (state.isProcessing) {
    if (orbStatus) orbStatus.textContent = "AI: PLANNING...";
    if (pill) { pill.textContent = "PLANNING"; pill.style.color = "#38bdf8"; }
  } else if (state.isListening) {
    btn.classList.add("listening");
    if (orbStatus) orbStatus.textContent = "MIC: LISTENING";
    if (pill) { pill.textContent = "RECORDING"; pill.style.color = "#ef4444"; }
  } else {
    if (orbStatus) orbStatus.textContent = "MIC: PUSH TO TALK";
    if (pill) { pill.textContent = "READY"; pill.style.color = "#10b981"; }
  }
}

function updateAuthorityUI() {
  const badge = document.getElementById("authority-badge");
  const btnAI = document.getElementById("btn-auth-ai");
  const btnManual = document.getElementById("btn-auth-manual");

  if (badge) {
    badge.className = `authority-badge authority-${state.currentAuthority.toLowerCase()}`;
    badge.textContent = `AUTH: ${state.currentAuthority}`;
  }

  if (btnAI) btnAI.classList.toggle("active-ai", state.currentAuthority === "AI");
  if (btnManual) btnManual.classList.toggle("active-teleop", state.currentAuthority === "MANUAL");
}

function logEvent(tag, message) {
  const logStream = document.getElementById("log-stream");
  if (!logStream) return;

  const entry = document.createElement("div");
  entry.className = "log-entry";

  const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const tagClass = tag.toLowerCase();

  entry.innerHTML = `
    <span class="log-timestamp">${time}</span>
    <span class="log-tag ${tagClass}">${tag}</span>
    <span class="log-msg">${escapeHtml(message)}</span>
  `;

  logStream.appendChild(entry);
  logStream.scrollTop = logStream.scrollHeight;
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ============================================================================
// Robot Real-Time State Polling
// ============================================================================
async function pollRobotState() {
  try {
    const response = await fetch("/api/v1/robot/state");
    if (response.ok) {
      const data = await response.json();
      
      const statusDot = document.getElementById("status-dot");
      const statusText = document.getElementById("status-text");
      const latencyVal = document.getElementById("telemetry-latency");
      const lastCmdVal = document.getElementById("telemetry-last-cmd");
      const authVal = document.getElementById("telemetry-authority");

      const isOnline = data.connection_status === "ONLINE";
      state.robotOnline = isOnline;

      if (statusDot) {
        statusDot.className = `status-dot ${isOnline ? 'online' : 'offline'}`;
      }
      if (statusText) {
        statusText.textContent = isOnline ? "ESP32 ONLINE" : "ESP32 OFFLINE";
      }
      if (latencyVal) {
        latencyVal.textContent = data.heartbeat_latency_ms ? `${data.heartbeat_latency_ms.toFixed(1)} ms` : "--";
      }
      if (lastCmdVal) {
        lastCmdVal.textContent = data.last_command_acknowledged || "None";
      }
      if (authVal) {
        authVal.textContent = data.control_authority || state.currentAuthority;
      }

      if (data.control_authority && data.control_authority !== state.currentAuthority) {
        state.currentAuthority = data.control_authority;
        updateAuthorityUI();
      }
    }
  } catch (e) {
    // Server down or offline
  }
}

// ============================================================================
// Initialization & Event Listeners
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
  // Resume AudioContext on any user interaction
  document.body.addEventListener("click", () => {
    audioCtrl.init();
    if (window.speechSynthesis) window.speechSynthesis.resume();
  }, { once: true });

  // 1. Voice Orb Click
  const orb = document.getElementById("voice-orb");
  if (orb) {
    orb.addEventListener("click", () => stt.toggle());
  }

  // 2. Mode buttons (Continuous vs Push to Talk)
  const modeContinuous = document.getElementById("mode-continuous");
  const modePush = document.getElementById("mode-push");
  if (modeContinuous && modePush) {
    modeContinuous.addEventListener("click", () => {
      modeContinuous.classList.add("active");
      modePush.classList.remove("active");
      state.continuousMode = true;
      stt.start();
    });
    modePush.addEventListener("click", () => {
      modePush.classList.add("active");
      modeContinuous.classList.remove("active");
      state.continuousMode = false;
      stt.stop();
    });
  }

  // 3. Fallback text input
  const textInput = document.getElementById("text-command-input");
  const sendBtn = document.getElementById("btn-send-text");
  const handleSendText = () => {
    if (!textInput) return;
    const val = textInput.value.trim();
    if (val) {
      processVoiceCommand(val);
      textInput.value = "";
    }
  };

  if (sendBtn) sendBtn.addEventListener("click", handleSendText);
  if (textInput) {
    textInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleSendText();
    });
  }

  // 4. Quick prompt chips
  document.querySelectorAll(".prompt-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt) processVoiceCommand(prompt);
    });
  });

  // 5. D-Pad Directional Controls
  const dpadActions = {
    "dpad-forward": "Move forward slowly",
    "dpad-backward": "Move backward slowly",
    "dpad-left": "Turn left",
    "dpad-right": "Turn right",
  };

  Object.entries(dpadActions).forEach(([id, direction]) => {
    const btn = document.getElementById(id);
    if (btn) {
      btn.addEventListener("click", () => {
        const naturalCommand = `${direction} for ${state.steps} step`;
        processVoiceCommand(naturalCommand);
      });
    }
  });

  const stopBtn = document.getElementById("dpad-stop");
  if (stopBtn) {
    stopBtn.addEventListener("click", () => handleEmergencyStop());
  }

  // 6. Keyboard Shortcuts (WASD / Arrows)
  window.addEventListener("keydown", (e) => {
    if (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA") {
      return;
    }

    if (e.key === "ArrowUp" || e.key === "w" || e.key === "W") {
      e.preventDefault();
      document.getElementById("dpad-forward")?.click();
    } else if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
      e.preventDefault();
      document.getElementById("dpad-backward")?.click();
    } else if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") {
      e.preventDefault();
      document.getElementById("dpad-left")?.click();
    } else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") {
      e.preventDefault();
      document.getElementById("dpad-right")?.click();
    } else if (e.key === " " || e.key === "Spacebar") {
      e.preventDefault();
      document.getElementById("dpad-stop")?.click();
    }
  });

  // 7. Speed profile selection
  document.querySelectorAll(".speed-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.speedProfile = btn.getAttribute("data-speed");
      state.speedPwm = parseInt(btn.getAttribute("data-pwm"), 10);
      logEvent("EXEC", `Speed profile set to: ${state.speedProfile.toUpperCase()} (${state.speedPwm} PWM)`);
    });
  });

  // 8. Step duration selection
  document.querySelectorAll(".step-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".step-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.steps = parseInt(btn.getAttribute("data-steps"), 10);
      logEvent("EXEC", `Step duration set to: ${state.steps} step(s)`);
    });
  });

  // 9. Quick Authority Switchers (AI vs MANUAL vs ESTOP)
  document.getElementById("btn-auth-ai")?.addEventListener("click", () => setAuthority("AI", true));
  document.getElementById("btn-auth-manual")?.addEventListener("click", () => setAuthority("MANUAL", true));
  document.getElementById("btn-auth-estop")?.addEventListener("click", () => handleEmergencyStop());

  // 10. Voice Settings & Test TTS button
  const voiceSelect = document.getElementById("voice-select");
  if (voiceSelect) {
    voiceSelect.addEventListener("change", (e) => {
      state.ttsVoice = state.availableVoices[e.target.value];
      tts.speak("Voice model selected.");
    });
  }

  const rateSlider = document.getElementById("tts-rate");
  if (rateSlider) {
    rateSlider.addEventListener("input", (e) => {
      state.ttsRate = parseFloat(e.target.value);
    });
  }

  const ttsToggle = document.getElementById("tts-toggle");
  if (ttsToggle) {
    ttsToggle.addEventListener("change", (e) => {
      state.ttsEnabled = e.target.checked;
    });
  }

  const testTtsBtn = document.getElementById("btn-test-tts");
  if (testTtsBtn) {
    testTtsBtn.addEventListener("click", () => {
      audioCtrl.init();
      tts.speak("IOFT Robot speech synthesizer online and operational.");
    });
  }

  // 11. Initial State Poll & Audio Device Discovery
  audioCtrl.listAudioDevices();
  pollRobotState();
  setInterval(pollRobotState, 1500);

  // 12. Microphone Test & Device Selector Listeners
  const testMicBtn = document.getElementById("btn-test-mic");
  if (testMicBtn) {
    testMicBtn.addEventListener("click", async () => {
      audioCtrl.init();
      const devSelect = document.getElementById("mic-device-select");
      const devId = devSelect ? devSelect.value : null;
      testMicBtn.textContent = "Listening...";
      testMicBtn.style.color = "#38bdf8";
      const ok = await audioCtrl.startMicVisualizer(devId);
      if (ok) {
        await audioCtrl.listAudioDevices();
        testMicBtn.textContent = "Active";
        testMicBtn.style.color = "#10b981";
        logEvent("STT", "Live mic test active — speak to see the VU meter bar jump!");
      } else {
        testMicBtn.textContent = "Error";
        testMicBtn.style.color = "#ef4444";
      }
    });
  }

  const micDevSelect = document.getElementById("mic-device-select");
  if (micDevSelect) {
    micDevSelect.addEventListener("change", (e) => {
      audioCtrl.startMicVisualizer(e.target.value);
      logEvent("STT", `Switched microphone input device`);
    });
  }

  const langSelect = document.getElementById("stt-lang-select");
  if (langSelect && stt.recognition) {
    langSelect.addEventListener("change", (e) => {
      stt.recognition.lang = e.target.value;
      logEvent("STT", `Speech recognition language set to: ${e.target.value}`);
    });
  }

  logEvent("SYS", "IOFT Robot AI Cockpit loaded. STT & TTS ready.");
});
