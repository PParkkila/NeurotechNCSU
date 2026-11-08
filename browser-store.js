// Session data lives in IndexedDB for this browser and origin.
(function () {
  const databaseName = "neuropawn-browser-sessions";
  const storeName = "sessions";
  const idPattern = /^[0-9a-f]{8}(?:-?[0-9a-f]{4}){3}-?[0-9a-f]{12}$/i;
  let opening;

  function storageError(error) {
    if (error?.name === "QuotaExceededError") {
      return new Error("Browser storage is full. Free up some space before saving another EEG read.");
    }
    return new Error("This browser could not access local session storage.", { cause: error });
  }

  function open() {
    opening ??= new Promise((resolve, reject) => {
      const request = indexedDB.open(databaseName, 1);
      let blocked = false;
      request.onupgradeneeded = () => request.result.createObjectStore(storeName, { keyPath: "id" });
      request.onblocked = () => {
        blocked = true;
        reject(new Error("Close other Shape Battle tabs and try again to open session storage."));
      };
      request.onerror = () => reject(storageError(request.error));
      request.onsuccess = () => {
        const database = request.result;
        if (blocked) {
          database.close();
          return;
        }
        database.onversionchange = () => {
          database.close();
          opening = undefined;
        };
        database.onclose = () => { opening = undefined; };
        resolve(database);
      };
    }).catch((error) => {
      opening = undefined;
      throw error;
    });
    return opening;
  }

  function validateId(id) {
    if (typeof id !== "string" || !idPattern.test(id)) {
      throw new Error("Invalid local EEG session ID.");
    }
  }

  function validateRecord(record, allowDraft = false) {
    validateId(record?.id);
    const revision = allowDraft ? record.revision ?? 0 : record.revision;
    if (!Number.isSafeInteger(revision) || revision < (allowDraft ? 0 : 1) ||
        typeof record.player !== "string" || !record.configuration ||
        !Array.isArray(record.schedule) || !Array.isArray(record.calibration) ||
        !Array.isArray(record.rounds)) {
      throw new Error("The local EEG session record is incomplete or invalid.");
    }
    return revision;
  }

  async function get(id) {
    validateId(id);
    const database = await open();
    return new Promise((resolve, reject) => {
      const transaction = database.transaction(storeName, "readonly");
      const request = transaction.objectStore(storeName).get(id);
      transaction.onabort = () => reject(storageError(transaction.error));
      transaction.oncomplete = () => {
        try {
          const record = request.result;
          if (record !== undefined) validateRecord(record);
          resolve(record ?? null);
        } catch (error) {
          reject(error);
        }
      };
    });
  }

  async function put(record) {
    const revision = validateRecord(record, true);
    const next = structuredClone({ ...record, revision: revision + 1 });
    const database = await open();
    return new Promise((resolve, reject) => {
      const transaction = database.transaction(storeName, "readwrite");
      const store = transaction.objectStore(storeName);
      const request = store.get(next.id);
      let conflict;
      transaction.onabort = () => reject(conflict ?? storageError(transaction.error));
      transaction.oncomplete = () => resolve(next);
      request.onsuccess = () => {
        const currentRevision = request.result === undefined ? 0 : request.result.revision;
        if (currentRevision !== revision) {
          conflict = new Error("This session changed in another tab. Reload and resume it before making more changes.");
          conflict.status = 409;
          transaction.abort();
          return;
        }
        store.put(next);
      };
    });
  }

  function download(record) {
    validateRecord(record);
    const json = JSON.stringify({ format: "neuropawn-web-session", version: 1, ...record },
      (_, value) => ArrayBuffer.isView(value) && !(value instanceof DataView) ? Array.from(value) : value);
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `neuropawn-session-${record.id}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  window.NPBrowserStore = { get, put, download };
})();
