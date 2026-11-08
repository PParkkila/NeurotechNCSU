// Welch/PCA features with three pairwise classifiers and bounded feedback learning.
(function () {
  const bands = [[4, 8], [8, 13], [13, 20], [20, 31]];
  const classCount = 3;
  // Indices follow NPShapes.LIST: circle, triangle, square.
  const pairs = [[0, 2], [1, 0], [2, 1]];
  const policyVersion = 3;
  const feedbackWeight = 0.25;
  const featureRadius = 3;
  // The appended bias adds one to the squared row norm.
  const coefficientStep = 0.12 / Math.hypot(featureRadius, 1);
  const kernels = new Map();

  function welchKernel(fs) {
    if (!kernels.has(fs)) {
      const window = Array.from({ length: fs }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / fs));
      const scale = 1 / (fs * window.reduce((sum, w) => sum + w * w, 0));
      const bins = bands.map(([low, high]) => Array.from({ length: high - low }, (_, index) => {
        const frequency = low + index;
        const angle = 2 * Math.PI * frequency / fs;
        return {
          real: window.map((weight, i) => weight * Math.cos(angle * i)),
          imag: window.map((weight, i) => weight * Math.sin(angle * i)),
          scale: frequency * 2 === fs ? scale : 2 * scale,
        };
      }));
      kernels.set(fs, bins);
    }
    return kernels.get(fs);
  }

  function logPowers(channel, fs, start) {
    const bins = welchKernel(fs);
    const hop = fs - Math.floor(fs / 2);
    const sums = bins.map(() => 0);
    let count = 0;
    for (let offset = start; offset + fs <= channel.length; offset += hop) {
      let mean = 0;
      for (let i = 0; i < fs; i++) mean += channel[offset + i];
      mean /= fs;
      for (let band = 0; band < bins.length; band++) {
        for (const { real, imag, scale } of bins[band]) {
          let re = 0, im = 0;
          for (let i = 0; i < fs; i++) {
            const value = channel[offset + i] - mean;
            re += real[i] * value;
            im += imag[i] * value;
          }
          sums[band] += (re * re + im * im) * scale;
        }
      }
      count++;
    }
    return sums.map((power) => Math.log(Math.max(power / count, 1e-12)));
  }

  function spectra(epoch, fs, start, allInputs) {
    if (!Array.isArray(epoch) || !epoch.length || !epoch.every((channel) =>
      channel?.length === epoch[0]?.length && channel.length >= start + fs && channel.every?.(Number.isFinite))) {
      throw new Error("EEG must contain equal-length channels with enough finite samples.");
    }
    const inputs = allInputs ? epoch : epoch.slice(0, 6);
    return inputs.flatMap((channel) => logPowers(channel, fs, start));
  }

  function features(imagery, fs, baseline, { skipSeconds = 1, allInputs = false } = {}) {
    if (!Number.isInteger(fs) || fs < 60) throw new Error("Invalid EEG sample rate.");
    const start = skipSeconds * fs;
    if (!Number.isInteger(start) || start < 0) throw new Error("Invalid EEG imagery delay.");
    const power = spectra(imagery, fs, start, allInputs);
    if (!baseline) return power;
    if (baseline.length !== imagery.length) throw new Error("Rest and imagery need the same EEG channels.");
    const rest = spectra(baseline, fs, 0, allInputs);
    return power.map((value, i) => value - rest[i]);
  }

  const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

  function projector(rows) {
    const dimensions = rows[0].length;
    const axes = Array.from(rows[0], (_, column) => rows.map((row) => row[column]));
    const means = axes.map((axis) => axis.reduce((a, b) => a + b, 0) / rows.length);
    const scales = axes.map((axis, j) => {
      const variance = axis.reduce((sum, value) => sum + (value - means[j]) ** 2, 0) / rows.length;
      const noise = rows.length * Number.EPSILON;
      return variance <= noise * variance + (noise * means[j]) ** 2 ? 1 : Math.sqrt(variance);
    });
    const scaled = axes.map((axis, j) => axis.map((value) => (value - means[j]) / scales[j]));
    const cov = scaled.map((axis) => scaled.map((other) => dot(axis, other) / rows.length));
    const basis = axes.map((_, i) => axes.map((_, j) => Number(i === j)));

    // Jacobi eigendecomposition of the small symmetric covariance matrix.
    for (let sweep = 0; sweep < 30; sweep++) {
      let changed = false;
      for (let p = 0; p < dimensions; p++) {
        for (let q = p + 1; q < dimensions; q++) {
          const pp = cov[p][p], qq = cov[q][q], pq = cov[p][q];
          if (Math.abs(pq) < 1e-12) continue;
          changed = true;
          const theta = 0.5 * Math.atan2(2 * pq, qq - pp);
          const c = Math.cos(theta), s = Math.sin(theta);
          for (let j = 0; j < dimensions; j++) {
            if (j !== p && j !== q) {
              const jp = cov[j][p], jq = cov[j][q];
              cov[j][p] = cov[p][j] = c * jp - s * jq;
              cov[j][q] = cov[q][j] = s * jp + c * jq;
            }
            const vp = basis[j][p], vq = basis[j][q];
            basis[j][p] = c * vp - s * vq;
            basis[j][q] = s * vp + c * vq;
          }
          cov[p][p] = c * c * pp - 2 * c * s * pq + s * s * qq;
          cov[q][q] = s * s * pp + 2 * c * s * pq + c * c * qq;
          cov[p][q] = cov[q][p] = 0;
        }
      }
      if (!changed) break;
    }
    const components = axes.map((_, i) => i).sort((a, b) => cov[b][b] - cov[a][a])
      .slice(0, 5).map((axis) => basis.map((row) => row[axis]));
    return (row) => {
      const scaledRow = row.map((value, i) => (value - means[i]) / scales[i]);
      const scores = components.map((axis) => dot(axis, scaledRow));
      const norm = Math.hypot(...scores);
      const scale = norm > featureRadius ? featureRadius / norm : 1;
      return [...scores.map((value) => value * scale), 1];
    };
  }

  const sigmoid = (value) => 1 / (1 + Math.exp(-value));
  const penalty = (index, width) => index === width - 1 ? 0.25 : 1;

  function loss(weights, rows) {
    let value = 0;
    for (const { row, target, weight } of rows) {
      const margin = dot(weights, row);
      // Stable logistic loss for either binary label.
      const wrongMargin = target ? -margin : margin;
      value += weight * (Math.max(wrongMargin, 0) + Math.log1p(Math.exp(-Math.abs(wrongMargin))));
    }
    return value + weights.reduce((sum, weight, i) => sum + 0.5 * penalty(i, weights.length) * weight ** 2, 0);
  }

  function gradient(weights, rows) {
    const result = weights.map((value, i) => penalty(i, weights.length) * value);
    for (const { row, target, weight } of rows) {
      const error = weight * (sigmoid(dot(weights, row)) - target);
      for (let i = 0; i < result.length; i++) result[i] += error * row[i];
    }
    return result;
  }

  function initialWeights(rows) {
    const width = rows[0].row.length;
    const step = 1 / (1 + 0.25 * rows.reduce((sum, { row, weight }) => sum + weight * dot(row, row), 0));
    let estimate = Array(width).fill(0), point = estimate;
    for (let iteration = 0; iteration < 300; iteration++) {
      const direction = gradient(point, rows);
      if (direction.every((value) => Math.abs(value) < 1e-7)) break;
      const next = point.map((value, i) => value - step * direction[i]);
      const momentum = iteration / (iteration + 3);
      point = next.map((value, i) => value + momentum * (value - estimate[i]));
      estimate = next;
    }
    return estimate;
  }

  function randomGenerator(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let bits = Math.imul(state ^ (state >>> 15), state | 1);
      bits ^= bits + Math.imul(bits ^ (bits >>> 7), bits | 61);
      return ((bits ^ (bits >>> 14)) >>> 0) / 4294967296;
    };
  }

  function boundedProposal(anchor, proposal) {
    const delta = proposal.map((value, i) => value - anchor[i]);
    const distance = Math.hypot(...delta);
    if (distance <= coefficientStep) return proposal;
    return anchor.map((value, i) => value + delta[i] * coefficientStep / distance);
  }

  function anneal(anchor, rows, seed) {
    const random = randomGenerator(seed);
    let current = anchor, currentLoss = loss(anchor, rows);
    const direction = gradient(anchor, rows);
    const norm = Math.hypot(...direction);
    if (norm < 1e-7) return anchor;
    // Start the search with a downhill proposal. Every proposal stays within the
    // same trust radius, including when the Metropolis step accepts a worse fit.
    const start = anchor.map((value, i) => value - coefficientStep * direction[i] / norm);
    const startLoss = loss(start, rows);
    const initialTemp = Math.max(0.002, Math.abs(currentLoss - startLoss));
    if (startLoss < currentLoss) {
      current = start;
      currentLoss = startLoss;
    }
    let best = current, bestLoss = currentLoss;
    for (let iteration = 0; iteration < 96; iteration++) {
      const progress = iteration / 95;
      const temperature = initialTemp * 0.01 ** progress;
      const step = coefficientStep * (0.6 * (1 - progress) + 0.05);
      const proposal = boundedProposal(anchor, current.map((value) => value + step * (2 * random() - 1)));
      const proposedLoss = loss(proposal, rows);
      if (proposedLoss < currentLoss || random() < Math.exp((currentLoss - proposedLoss) / temperature)) {
        current = proposal;
        currentLoss = proposedLoss;
      }
      if (proposedLoss < bestLoss) {
        best = proposal;
        bestLoss = proposedLoss;
      }
    }
    return best;
  }

  function couple(margins) {
    // Project pairwise log odds onto one set of centered class logits. For the
    // complete, equally weighted graph the least-squares score is the signed
    // sum of outgoing edges divided by the number of classes.
    const scores = Array(classCount).fill(0);
    for (const [index, [first, second]] of pairs.entries()) {
      scores[first] += margins[index] / classCount;
      scores[second] -= margins[index] / classCount;
    }
    const peak = Math.max(...scores);
    const probabilities = scores.map((score) => Math.exp(score - peak));
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    return probabilities.map((value) => value / total);
  }

  const best = (scores) => scores.indexOf(Math.max(...scores));

  function train(calibration, feedback = []) {
    const dimensions = calibration[0]?.features?.length;
    const validRow = (row) => row?.length === dimensions && row.every?.(Number.isFinite);
    const validLabel = (label) => Number.isInteger(label) && label >= 0 && label < classCount;
    if (!dimensions || new Set(calibration.map((item) => item.label)).size !== classCount ||
        !calibration.every((item) => validLabel(item.label) && validRow(item.features)) ||
        !feedback.every((item) => validLabel(item.actual) && validRow(item.features))) {
      throw new Error("The decoder needs valid calibration reads covering all three shapes.");
    }
    const project = projector(calibration.map((item) => item.features));
    const startingRows = calibration.map((item) => ({ row: project(item.features), label: item.label, weight: 1 }));
    const replay = feedback.map((item) => ({ row: project(item.features), label: item.actual, weight: feedbackWeight }));
    const last = feedback.at(-1);
    const seed = [], learner = [];
    for (const [index, [first, second]] of pairs.entries()) {
      function pairRows(rows) {
        return rows.filter(({ label }) => label === first || label === second)
          .map(({ row, label, weight }) => ({ row, target: Number(label === first), weight }));
      }
      const initialRows = pairRows(startingRows);
      const initial = initialWeights(initialRows);
      seed.push(initial);
      const saved = last?.prediction?.policy_version === policyVersion ? last.prediction.learner?.[index] : null;
      const canResume = Array.isArray(saved) && saved.length === initial.length && saved.every(Number.isFinite);
      const anchor = canResume ? saved : initial;
      const shouldLearn = last && (!canResume || last.actual === first || last.actual === second);
      learner.push(shouldLearn
        ? anneal(anchor, [...initialRows, ...pairRows(replay)], 9741 + 997 * feedback.length + index)
        : anchor);
    }
    return {
      predict(row) {
        if (!validRow(row)) throw new Error("The EEG features do not match this calibration.");
        const projected = project(row);
        const base = couple(seed.map((weights) => dot(weights, projected)));
        const margins = learner.map((weights) => dot(weights, projected));
        const raw = couple(margins);
        const move = best(raw), seedMove = best(base);
        return {
          move, confidence: raw[move], raw_probabilities: raw,
          seed_move: seedMove, seed_confidence: base[seedMove],
          pairwise: pairs.map((shapes, index) => ({ shapes: [...shapes], probability: sigmoid(margins[index]) })),
          examples: feedback.length, using_player: feedback.length > 0,
          learner: learner.map((weights) => [...weights]), policy_version: policyVersion,
        };
      },
    };
  }

  function fit(calibration, feedback = []) {
    if (![9, 20, 150].includes(calibration.length)) {
      throw new Error("Finish all calibration reads before playing.");
    }
    return train(calibration, feedback);
  }

  function validateHoldout(training, heldOut) {
    const model = train(training);
    return heldOut.map(({ label, features }) => ({
      actual: label, raw_probabilities: model.predict(features).raw_probabilities,
    }));
  }

  function crossValidation(calibration) {
    return calibration.map((item, index) => {
      const training = calibration.filter((_, other) => other !== index);
      const { raw_probabilities } = train(training).predict(item.features);
      return { actual: item.label, raw_probabilities };
    });
  }

  window.NPBrowserDecoder = { features, fit, validateHoldout, crossValidation };
})();
