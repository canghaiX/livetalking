'use strict';

// Area resampling keeps fractional source samples across render quanta.
class PCMCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.weight = 0;
    this.sum = 0;
    this.samples = new Int16Array(1600);
    this.length = 0;
    this.stopped = false;
    this.port.onmessage = event => {
      if (event.data.type === 'flush') {
        this.stopped = true;
        this.send();
        this.port.postMessage({type: 'flushed'});
      }
    };
  }

  send() {
    if (!this.length) return;
    const buffer = this.samples.slice(0, this.length).buffer;
    this.port.postMessage({type: 'pcm', data: buffer}, [buffer]);
    this.length = 0;
  }

  process(inputs) {
    if (this.stopped || !inputs[0]?.[0]) return true;
    for (const value of inputs[0][0]) {
      let remaining = 1;
      while (remaining > 1e-8) {
        const take = Math.min(remaining, this.ratio - this.weight);
        this.sum += value * take;
        this.weight += take;
        remaining -= take;
        if (this.weight >= this.ratio - 1e-8) {
          const sample = Math.max(-1, Math.min(1, this.sum / this.ratio));
          this.samples[this.length++] = Math.round(sample * (sample < 0 ? 32768 : 32767));
          this.weight = 0;
          this.sum = 0;
          if (this.length === this.samples.length) this.send();
        }
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PCMCapture);
