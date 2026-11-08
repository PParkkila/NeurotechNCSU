// The hosted game uses the same view as the Python bridge. EEG stays on this device.
(function () {
  const SHAPES = window.NPShapes.LIST;
  const REST_SECONDS = 2;
  const CUE_SECONDS = 1;
  const IMAGERY_SECONDS = 5;
  const DATASET_FORMAT = "neuropawn-rps-dataset";
  const DATASET_BLOCKS = 10;
  const TRIALS_PER_BLOCK = 15;
  const TRIAL_COUNT = DATASET_BLOCKS * TRIALS_PER_BLOCK;
  const TRAINING_BLOCKS = DATASET_BLOCKS - 2;
  const MAX_DATASET_BYTES = 32 * 1024 * 1024;
  const byte = new Uint8Array(1);
  const now = () => new Date().toISOString();

  function randomBelow(limit) {
    const ceiling = 256 - (256 % limit);
    do { crypto.getRandomValues(byte); } while (byte[0] >= ceiling);
    return byte[0] % limit;
  }

  function schedule() {
    return Array.from({ length: DATASET_BLOCKS }, () => {
      const block = SHAPES.flatMap((_, index) => Array(TRIALS_PER_BLOCK / SHAPES.length).fill(index));
      for (let i = block.length - 1; i > 0; i--) {
        const j = randomBelow(i + 1);
        [block[i], block[j]] = [block[j], block[i]];
      }
      return block;
    }).flat();
  }

  function shapeCounts(confusion) {
    return confusion.map((row, index) => {
      const actual = row.reduce((sum, count) => sum + count, 0);
      const predicted = confusion.reduce((sum, other) => sum + other[index], 0);
      return { shape: SHAPES[index], hits: row[index], actual, predicted };
    });
  }

  function metrics(per_shape, total) {
    const correct = per_shape.reduce((sum, shape) => sum + shape.hits, 0);
    function macro(field) {
      const supported = per_shape.filter((shape) => shape[field] > 0);
      return supported.length
        ? supported.reduce((sum, shape) => sum + shape.hits / shape[field], 0) / supported.length : null;
    }
    return {
      total, correct, accuracy: total ? correct / total : null,
      macro_recall: macro("actual"), macro_precision: macro("predicted"),
    };
  }

  function state(record) {
    const completed = record.calibration.length;
    const last = record.rounds.at(-1);
    const current = last?.status === "discarded" ? null : last;
    const total = record.schedule.length;
    const phase = completed < total ? "calibration" : current?.status ?? "ready";
    const round = current ? {
      id: current.id,
      predicted: SHAPES[current.prediction.move],
      score: current.prediction.confidence,
      model: current.prediction.using_player ? "player" : "seed",
      examples: current.prediction.examples,
      pairwise: current.prediction.pairwise,
      confidenceReads: current.prediction.confidence_calibration_reads,
      confidenceFeedback: current.prediction.confidence_feedback_reads,
      clippedInput: current.clippedInput ?? null,
      confirmed: SHAPES[current.actual] ?? null,
      computer: phase === "result" ? SHAPES[current.computer] : null,
    } : null;

    const confusion = SHAPES.map(() => Array(SHAPES.length).fill(0));
    const history = [], results = [];
    let lastConfirmed = null;
    for (const item of record.rounds) {
      if (item.actual === null) continue;
      confusion[item.actual][item.prediction.move]++;
      history.push({ round: history.length + 1, ...metrics(shapeCounts(confusion), history.length + 1) });
      lastConfirmed = { predicted: SHAPES[item.prediction.move], actual: SHAPES[item.actual] };
      if (item.status === "result") {
        results.push({ player: SHAPES[item.actual], computer: SHAPES[item.computer] });
      }
    }
    const perShape = shapeCounts(confusion);
    const dataset = record.dataset;
    return {
      id: record.id, player: record.player, phase, round, results,
      calibration: { completed, total, prompt: SHAPES[record.schedule[completed]] ?? null, rapidFire: record.rapidFire === true },
      dataset: dataset ? {
        holdoutTotal: dataset.heldOut.length,
        holdoutCorrect: dataset.heldOut.filter(({ actual, raw_probabilities: scores }) =>
          scores.indexOf(Math.max(...scores)) === actual).length,
      } : null,
      metrics: { ...metrics(perShape, history.length), history,
        per_shape: perShape, last_confirmed: lastConfirmed },
    };
  }

  function configuration(board) {
    return { fs: board.fs, inputs: [...board.inputs], gain: board.gain, imu: !!board.imu };
  }

  function matchingConfiguration(saved, current) {
    return saved?.fs === current.fs && saved.gain === current.gain && saved.imu === current.imu &&
      Array.isArray(saved.inputs) && saved.inputs.length === current.inputs.length &&
      saved.inputs.every((input, index) => input === current.inputs[index]);
  }

  async function readDataset(file, setup) {
    if (!file || file.size < 1 || file.size > MAX_DATASET_BYTES) {
      throw new Error("Choose a dataset JSON file of up to 32 MB.");
    }
    let data;
    try {
      data = JSON.parse(await file.text());
    } catch {
      throw new Error("The dataset file could not be read as JSON.");
    }
    if (data?.format !== DATASET_FORMAT || data.version !== 1 || !Array.isArray(data.trials) ||
        data.trials.length !== TRIAL_COUNT) {
      throw new Error("Choose a 150-trial NeuroPawn dataset exported by export-rps150.");
    }
    if (!matchingConfiguration(data.configuration, setup)) {
      throw new Error("The dataset uses different board settings. Reconnect with its input order, gain, and board variant.");
    }

    function validEpoch(epoch, seconds) {
      return Array.isArray(epoch) && epoch.length === setup.inputs.length && epoch.every((channel) =>
        Array.isArray(channel) && channel.length === seconds * setup.fs && channel.every(Number.isFinite));
    }

    for (let offset = 0; offset < data.trials.length; offset += TRIALS_PER_BLOCK) {
      const block = 1 + offset / TRIALS_PER_BLOCK;
      const counts = SHAPES.map(() => 0);
      for (const trial of data.trials.slice(offset, offset + TRIALS_PER_BLOCK)) {
        if (trial?.block !== block || !Number.isInteger(trial.label) || SHAPES[trial.label] === undefined ||
            !validEpoch(trial.baseline, REST_SECONDS) || !validEpoch(trial.eeg, IMAGERY_SECONDS - 1)) {
          throw new Error(`Dataset block ${block} contains an invalid trial.`);
        }
        counts[trial.label]++;
      }
      if (counts.some((count) => count !== TRIALS_PER_BLOCK / SHAPES.length)) {
        throw new Error(`Dataset block ${block} needs five trials of each shape.`);
      }
    }
    return data.trials;
  }

  function create() {
    const store = window.NPBrowserStore, decoder = window.NPBrowserDecoder;
    const confidence = window.NPBrowserConfidence;
    let board = null, trainedSeed;

    function activeBoard(record) {
      if (!board) throw new Error("Reconnect NeuroPawn before starting a read.");
      if (record && !matchingConfiguration(record.configuration, configuration(board))) {
        throw new Error("This session uses different board settings. Reconnect with the original settings or start new training.");
      }
      return board;
    }

    async function load(id) {
      const record = await store.get(id);
      if (!record) {
        const error = new Error("Session not found in this browser. Start new training.");
        error.status = 404;
        throw error;
      }
      if (record.version !== 1 || !Array.isArray(record.calibration) || !Array.isArray(record.rounds)) {
        throw new Error("This browser session has an unsupported format.");
      }
      return record;
    }

    async function save(record) {
      record.updatedAt = now();
      return state(await store.put(record));
    }

    async function finishTraining(record) {
      const split = TRAINING_BLOCKS * TRIALS_PER_BLOCK;
      const heldOut = decoder.validateHoldout(record.calibration.slice(0, split), record.calibration.slice(split));
      const model = decoder.fit(record.calibration);
      record.dataset = { format: DATASET_FORMAT, version: 1, heldOutBlocks: [TRAINING_BLOCKS + 1, DATASET_BLOCKS], heldOut };
      const ready = await save(record);
      trainedSeed = { id: record.id, model };
      return ready;
    }

    return {
      bridge: {
        rest_seconds: REST_SECONDS, imagery_seconds: IMAGERY_SECONDS, cue_seconds: CUE_SECONDS,
        calibration_count: TRIAL_COUNT,
      },
      setBoard(device) { board = device; },
      async getSession(id) { return state(await load(id)); },
      async downloadSession(id) { return store.download(await load(id)); },

      async importDataset(file, player) {
        const device = activeBoard();
        const setup = configuration(device);
        const trials = await readDataset(file, setup);
        if (device !== board) throw new Error("NeuroPawn disconnected during import. Reconnect and try again.");
        // Subtract paired resting log power to account for acquisition-scale differences.
        const calibration = trials.map(({ label, block, baseline, eeg }) => ({
          label, block, baseline, eeg,
          features: decoder.features(eeg, setup.fs, baseline, { skipSeconds: 0, allInputs: true }),
        }));
        return finishTraining({
          version: 1, id: crypto.randomUUID(), player, shapes: SHAPES, configuration: setup, createdAt: now(),
          schedule: calibration.map(({ label }) => label), calibration, rounds: [],
        });
      },

      async sessionCapture(id, question, onPhase) {
        const record = await load(id);
        const device = activeBoard(record);
        const completed = record.calibration.length;
        const last = record.rounds.at(-1);
        const calibrating = completed < record.schedule.length;
        if (calibrating ? question !== completed : question !== null || ["review", "confirmed"].includes(last?.status)) {
          throw new Error("The current prompt or round changed. Refresh the session and try again.");
        }

        const prompted = calibrating && record.rapidFire;
        const rest = await device.capture(REST_SECONDS, { onStart: () => onPhase("rest") });
        if (prompted) {
          onPhase("cue");
          await new Promise((resolve) => setTimeout(resolve, CUE_SECONDS * 1000));
          if (device !== board) throw new Error("NeuroPawn disconnected during the read. Reconnect and try again.");
        }
        const baseline = rest.samples;
        const { samples: eeg, clippedInput } = await device.capture(IMAGERY_SECONDS, {
          onStart: () => onPhase("go"), clippedInput: rest.clippedInput,
        });
        onPhase("decoding");
        if (device !== board) throw new Error("NeuroPawn disconnected during the read. Reconnect and try again.");
        const allInputs = Boolean(record.rapidFire || record.dataset);
        const ignoredChannel = device.inputs.indexOf(clippedInput);
        const features = decoder.features(eeg, device.fs, allInputs ? baseline : undefined, { allInputs, ignoredChannel });
        const trial = { baseline, eeg, features, capturedAt: now() };
        if (clippedInput !== null) trial.clippedInput = clippedInput;
        if (calibrating) {
          trial.label = record.schedule[completed];
          if (prompted) trial.block = 1 + Math.floor(completed / TRIALS_PER_BLOCK);
          record.calibration.push(trial);
          if (prompted && record.calibration.length === record.schedule.length) return finishTraining(record);
        } else {
          const feedback = record.rounds.filter((round) => round.actual !== null);
          const model = trainedSeed?.id === id && !feedback.length
            ? trainedSeed.model : decoder.fit(record.calibration, feedback);
          const raw = model.predict(features);
          const prediction = confidence.predict(raw, record.calibration, feedback, record.dataset?.heldOut);
          record.rounds.push({
            ...trial, id: crypto.randomUUID(), status: "review", actual: null,
            computer: randomBelow(SHAPES.length), prediction,
          });
        }
        return save(record);
      },

      async sessionAction(action, data) {
        if (action === "session") {
          return save({
            version: 1, id: crypto.randomUUID(), player: data.player, shapes: SHAPES,
            configuration: configuration(activeBoard()), schedule: schedule(),
            rapidFire: true, calibration: [], rounds: [], createdAt: now(),
          });
        }

        const record = await load(data.session);
        const round = record.rounds.at(-1);
        if (!round || round.id !== data.id) throw new Error("This round is no longer current.");
        switch (action) {
          case "confirm": {
            const actual = SHAPES.indexOf(data.shape);
            if (actual < 0) throw new Error("Choose circle, triangle, or square.");
            if (round.actual === actual) return state(record);
            if (round.status !== "review") throw new Error("Only a pending prediction can be confirmed.");
            Object.assign(round, { actual, status: "confirmed", confirmedAt: now() });
            break;
          }
          case "battle":
            if (round.status === "result") return state(record);
            if (round.status !== "confirmed") throw new Error("Confirm your shape before battling.");
            Object.assign(round, { status: "result", revealedAt: now() });
            break;
          case "discard":
            if (round.status === "discarded") return state(record);
            if (round.status !== "review") throw new Error("Only an unconfirmed prediction can be discarded.");
            Object.assign(round, { status: "discarded", discardedAt: now() });
            break;
          default:
            throw new Error("Unknown game action.");
        }
        return save(record);
      },
    };
  }

  window.NPBrowserGame = { create };
})();
