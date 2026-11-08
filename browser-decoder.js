// Browser equivalent of the Python game's Welch/PCA/logistic decoder.
(function () {
  const bands = [[4, 8], [8, 13], [13, 20], [20, 31]];
  const classCount = 3;
  const calibrationCount = 20;
  const policyVersion = 2;
  const blendPrior = 48;
  const blendStep = 0.05;
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

  function spectra(epoch, fs, start) {
    if (!Array.isArray(epoch) || !epoch.length || !epoch.every((channel) =>
      channel?.length === epoch[0]?.length && channel.length >= start + fs && channel.every?.(Number.isFinite))) {
      throw new Error("EEG must contain equal-length channels with enough finite samples.");
    }
    return epoch.slice(0, 6).flatMap((channel) => logPowers(channel, fs, start));
  }

  function features(imagery, fs, baseline) {
    if (!Number.isInteger(fs) || fs < 60) throw new Error("Invalid EEG sample rate.");
    const power = spectra(imagery, fs, fs);
    if (!baseline) return power;
    if (baseline.length !== imagery.length) throw new Error("Rest and imagery need the same EEG channels.");
    const rest = spectra(baseline, fs, 0);
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
      return [...components.map((axis) => dot(axis, scaledRow)), 1];
    };
  }

  function probabilities(coefficients, row, output) {
    for (let c = 0; c < classCount; c++) {
      let value = 0;
      for (let j = 0; j < row.length; j++) value += coefficients[c * row.length + j] * row[j];
      output[c] = value;
    }
    const max = Math.max(...output);
    let total = 0;
    for (let c = 0; c < classCount; c++) {
      output[c] = Math.exp(output[c] - max);
      total += output[c];
    }
    for (let c = 0; c < classCount; c++) output[c] /= total;
    return output;
  }

  function classifier(rows, labels, weights) {
    const width = rows[0].length;
    const size = classCount * width;
    const gradient = new Float64Array(size);
    const point = new Float64Array(size);
    const solution = new Float64Array(size);
    const scores = new Float64Array(classCount);
    // A Hessian upper bound gives a safe step for accelerated gradient descent.
    const step = 1 / (1 + 0.5 * rows.reduce((sum, row, i) => sum + weights[i] * dot(row, row), 0));
    for (let iteration = 0; iteration < 500; iteration++) {
      for (let j = 0; j < size; j++) gradient[j] = j % width === width - 1 ? 0 : point[j];
      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        probabilities(point, row, scores);
        for (let c = 0; c < classCount; c++) {
          const error = weights[i] * (scores[c] - Number(labels[i] === c));
          for (let j = 0; j < width; j++) gradient[c * width + j] += error * row[j];
        }
      }
      const momentum = iteration / (iteration + 3);
      for (let j = 0; j < size; j++) {
        const next = point[j] - step * gradient[j];
        point[j] = next + momentum * (next - solution[j]);
        solution[j] = next;
      }
      if (gradient.every((value) => Math.abs(value) < 1e-6)) break;
    }
    return (row) => probabilities(solution, row, new Float64Array(classCount));
  }

  function feedbackBlend(feedback) {
    const counts = Array(classCount).fill(0);
    const gains = Array(classCount).fill(0);
    for (const { actual, prediction } of feedback) {
      if (prediction?.policy_version !== policyVersion) continue;
      counts[actual]++;
      gains[actual] += Number(prediction.candidate_move === actual) - Number(prediction.seed_move === actual);
    }
    const count = counts.reduce((a, b) => a + b, 0);
    const candidate = count / (count + blendPrior);
    const eligible = count >= 12 && Math.min(...counts) >= 3 && gains.reduce((a, b) => a + b, 0) >= 2 &&
      gains.reduce((sum, gain, c) => sum + gain / counts[c], 0) >= 0.3 - 1e-12;
    const target = eligible ? candidate : 0;
    const previous = feedback.at(-1)?.prediction?.player_weight ?? 0;
    return { candidate, weight: Math.min(Math.max(target, previous - blendStep), previous + blendStep) };
  }

  function best(scores) {
    return scores.indexOf(Math.max(...scores));
  }

  function fit(calibration, feedback = []) {
    const dimensions = calibration[0]?.features?.length;
    const validRow = (row) => row?.length === dimensions && row.every?.(Number.isFinite);
    const validLabel = (label) => Number.isInteger(label) && label >= 0 && label < classCount;
    const seedLabels = calibration.map((item) => item.label);
    if (calibration.length !== calibrationCount || !dimensions || new Set(seedLabels).size !== classCount ||
        !calibration.every((item) => validLabel(item.label) && validRow(item.features)) ||
        !feedback.every((item) => validLabel(item.actual) && validRow(item.features))) {
      throw new Error("The decoder needs 20 valid calibration reads covering all three shapes.");
    }
    const project = projector(calibration.map((item) => item.features));
    const seedRows = calibration.map((item) => project(item.features));
    const seed = classifier(seedRows, seedLabels, seedLabels.map(() => 1));
    let player = seed;
    if (feedback.length) {
      const labels = [...seedLabels, ...feedback.map((item) => item.actual)];
      const counts = Array(classCount).fill(0);
      for (const label of seedLabels) counts[label]++;
      const weights = [...seedLabels.map(() => 18 / calibrationCount), ...feedback.map(() => 0.25)];
      const mass = Array(classCount).fill(0);
      labels.forEach((label, i) => { mass[label] += weights[i]; });
      const adjusted = weights.map((weight, i) => weight * counts[labels[i]] / mass[labels[i]]);
      player = classifier([...seedRows, ...feedback.map((item) => project(item.features))], labels, adjusted);
    }
    const blend = feedbackBlend(feedback);
    return {
      predict(row) {
        if (!validRow(row)) throw new Error("The EEG features do not match this calibration.");
        const projected = project(row);
        const base = seed(projected);
        const adapted = player(projected);
        const seedMove = best(base);
        const candidate = base.map((score, c) => score + blend.candidate * (adapted[c] - score));
        const displayed = base.map((score, c) => score + blend.weight * (adapted[c] - score));
        const candidateMove = best(candidate), move = best(displayed);
        return {
          move, confidence: displayed[move], seed_move: seedMove, seed_confidence: base[seedMove],
          candidate_move: candidateMove, candidate_confidence: candidate[candidateMove],
          examples: feedback.length, using_player: blend.weight > 0, player_weight: blend.weight,
          policy_version: policyVersion,
        };
      },
    };
  }

  window.NPBrowserDecoder = { features, fit };
})();
