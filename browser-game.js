// The hosted game uses the same view as the Python bridge. EEG stays on this device.
(function () {
  const SHAPES = window.NPShapes.LIST;
  const CALIBRATION_PER_SHAPE = 3;
  const REST_SECONDS = 2;
  const IMAGERY_SECONDS = 5;
  const byte = new Uint8Array(1);
  const now = () => new Date().toISOString();

  function randomBelow(limit) {
    const ceiling = 256 - (256 % limit);
    do { crypto.getRandomValues(byte); } while (byte[0] >= ceiling);
    return byte[0] % limit;
  }

  function schedule() {
    const prompts = SHAPES.flatMap((_, index) => Array(CALIBRATION_PER_SHAPE).fill(index));
    for (let i = prompts.length - 1; i > 0; i--) {
      const j = randomBelow(i + 1);
      [prompts[i], prompts[j]] = [prompts[j], prompts[i]];
    }
    return prompts;
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
    return {
      id: record.id, player: record.player, phase, round, results,
      calibration: { completed, total, prompt: SHAPES[record.schedule[completed]] ?? null },
      metrics: { ...metrics(perShape, history.length), history,
        per_shape: perShape, last_confirmed: lastConfirmed },
    };
  }

  function configuration(board) {
    return { fs: board.fs, inputs: [...board.inputs], gain: board.gain, imu: !!board.imu };
  }

  function create() {
    const store = window.NPBrowserStore, decoder = window.NPBrowserDecoder;
    const confidence = window.NPBrowserConfidence;
    let board = null;

    function activeBoard(record) {
      if (!board) throw new Error("Reconnect NeuroPawn before starting a read.");
      if (record && JSON.stringify(record.configuration) !== JSON.stringify(configuration(board))) {
        throw new Error("This session uses different board settings. Reconnect with the original settings or start a new calibration.");
      }
      return board;
    }

    async function load(id) {
      const record = await store.get(id);
      if (!record) {
        const error = new Error("Session not found in this browser. Start a new calibration.");
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

    return {
      bridge: {
        rest_seconds: REST_SECONDS, imagery_seconds: IMAGERY_SECONDS,
        calibration_count: CALIBRATION_PER_SHAPE * SHAPES.length,
      },
      setBoard(device) { board = device; },
      async getSession(id) { return state(await load(id)); },
      async downloadSession(id) { return store.download(await load(id)); },

      async sessionCapture(id, question, onPhase) {
        const record = await load(id);
        const device = activeBoard(record);
        const completed = record.calibration.length;
        const last = record.rounds.at(-1);
        const calibrating = completed < record.schedule.length;
        if (calibrating ? question !== completed : question !== null || ["review", "confirmed"].includes(last?.status)) {
          throw new Error("The current prompt or round changed. Refresh the session and try again.");
        }

        const baseline = await device.capture(REST_SECONDS, { onStart: () => onPhase("rest") });
        const eeg = await device.capture(IMAGERY_SECONDS, { onStart: () => onPhase("go") });
        onPhase("decoding");
        if (device !== board) throw new Error("NeuroPawn disconnected during the read. Reconnect and try again.");
        const features = decoder.features(eeg, device.fs);
        const trial = { baseline, eeg, features, capturedAt: now() };
        if (calibrating) {
          record.calibration.push({ ...trial, label: record.schedule[completed] });
        } else {
          const feedback = record.rounds.filter((round) => round.actual !== null);
          const raw = decoder.fit(record.calibration, feedback).predict(features);
          const prediction = confidence.predict(raw, record.calibration, feedback);
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
            calibration: [], rounds: [], createdAt: now(),
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
