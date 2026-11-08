// Temperature scaling uses held-out or pre-feedback probabilities only.
(function () {
  const shapeCount = 3;
  const prior = 0.5;
  const priorStrength = 4;
  const minimum = 0.05;
  const maximum = 1.25;
  const promptedWeight = 0.5;

  function logScores(probabilities) {
    if (!Array.isArray(probabilities) || probabilities.length !== shapeCount) return null;
    const values = Array.from(probabilities);
    if (values.some((value) => !Number.isFinite(value) || value < 0 || value > 1) ||
        !values.some((value) => value > 0)) return null;
    const scores = values.map((value) => Math.log(Math.max(value, 1e-12)));
    const largest = Math.max(...scores);
    return scores.map((score) => score - largest);
  }

  function normalize(scores, inverseTemperature) {
    const powered = scores.map((score) => Math.exp(score * inverseTemperature));
    const total = powered.reduce((sum, value) => sum + value, 0);
    return powered.map((value) => value / total);
  }

  function evidence(records, probabilities, weight) {
    const rows = [];
    for (const record of records) {
      const actual = record.actual;
      const scores = logScores(probabilities(record));
      if (scores && Number.isInteger(actual) && actual >= 0 && actual < shapeCount) {
        rows.push({ actual, scores, weight });
      }
    }
    return rows;
  }

  function fit(rows) {
    // Penalized log loss is convex in the positive inverse temperature.
    let low = minimum, high = maximum;
    for (let iteration = 0; iteration < 24; iteration++) {
      const value = (low + high) / 2;
      let gradient = priorStrength * (value - prior);
      for (const { scores, actual, weight } of rows) {
        const adjusted = normalize(scores, value);
        const expected = scores.reduce((sum, score, index) => sum + score * adjusted[index], 0);
        gradient += weight * (expected - scores[actual]);
      }
      if (gradient > 0) high = value;
      else low = value;
    }
    return (low + high) / 2;
  }

  function predict(rawPrediction, calibration, priorConfirmed = []) {
    const scores = logScores(rawPrediction.raw_probabilities);
    if (!scores || !Number.isInteger(rawPrediction.move) || scores[rawPrediction.move] === undefined) {
      throw new Error("The model returned invalid shape probabilities.");
    }
    const heldOut = window.NPBrowserDecoder.crossValidation(calibration);
    // Prompted reads use smaller weight than privately chosen gameplay reads.
    const initial = evidence(heldOut, (record) => record.raw_probabilities, promptedWeight);
    const later = evidence(priorConfirmed, (record) => record.prediction?.raw_probabilities, 1);
    const inverseTemperature = fit([...initial, ...later]);
    // Give the uniform prior two equivalent reads so the first few observations
    // cannot turn a sharp model score into an equally sharp confidence estimate.
    const priorMix = 2 / (3 + initial.length * promptedWeight + later.length);
    const probabilities = normalize(scores, inverseTemperature)
      .map((value) => (1 - priorMix) * value + priorMix / shapeCount);
    return {
      ...rawPrediction,
      probabilities,
      confidence: probabilities[rawPrediction.move],
      inverse_temperature: inverseTemperature,
      confidence_prior_mix: priorMix,
      confidence_calibration_reads: initial.length,
      confidence_feedback_reads: later.length,
    };
  }

  window.NPBrowserConfidence = { predict };
})();
