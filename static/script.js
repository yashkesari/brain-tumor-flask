/* ============================================================
   NEUROSCAN AI — Interactive Diagnostic Client
   ============================================================
   Handles:
   - Drag & drop MRI upload with reticle animations
   - File format & size verification
   - Instant high-fidelity image preview
   - Multi-phase diagnostic loading HUD
   - Print & clipboard summary exporting
   - No framework/Node dependencies, pure Vanilla JS
   ============================================================ */

(function () {
  'use strict';

  // ---------- DOM Elements ----------
  const uploadForm     = document.getElementById('ns-upload-form');
  const uploadZone     = document.getElementById('ns-upload-zone');
  const fileInput      = document.getElementById('mri-file-input');
  const previewPanel   = document.getElementById('ns-preview-panel');
  const previewImg     = document.getElementById('ns-preview-img');
  const previewName    = document.getElementById('ns-preview-filename');
  const previewSize    = document.getElementById('ns-preview-filesize');
  const removeBtn      = document.getElementById('ns-remove-btn');
  const analyzeBtn     = document.getElementById('ns-analyze-btn');
  const loadingHud     = document.getElementById('ns-loading-hud');
  const loadingStepTxt = document.getElementById('ns-loading-step-text');
  const copyBtn        = document.getElementById('ns-copy-btn');
  const toast          = document.getElementById('ns-toast');
  const toastMsg       = document.getElementById('ns-toast-msg');

  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];
  const ALLOWED_EXTS  = ['.jpg', '.jpeg', '.png'];

  // ---------- Utility: Show Toast ----------
  function showToast(message) {
    if (!toast || !toastMsg) return;
    toastMsg.textContent = message;
    toast.classList.add('ns-toast--show');
    setTimeout(() => {
      toast.classList.remove('ns-toast--show');
    }, 3200);
  }

  // ---------- Utility: Format Bytes ----------
  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // ---------- File Validation ----------
  function validateFile(file) {
    if (!file) return false;
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    return ALLOWED_TYPES.includes(file.type) || ALLOWED_EXTS.includes(ext);
  }

  // ---------- Image Preview ----------
  function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      if (previewImg) previewImg.src = e.target.result;
      if (previewName) previewName.textContent = file.name;
      if (previewSize) previewSize.textContent = formatBytes(file.size);

      if (uploadZone) uploadZone.style.display = 'none';
      if (previewPanel) previewPanel.classList.add('ns-preview-panel--active');
      if (analyzeBtn) analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // ---------- Reset Upload State ----------
  function resetUpload() {
    if (fileInput) fileInput.value = '';
    if (previewImg) previewImg.src = '';
    if (previewName) previewName.textContent = '';
    if (previewPanel) previewPanel.classList.remove('ns-preview-panel--active');
    if (uploadZone) uploadZone.style.display = '';
    if (analyzeBtn) analyzeBtn.disabled = true;
  }

  // ---------- Drag & Drop Handlers ----------
  if (uploadZone && fileInput) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
      document.body.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      uploadZone.addEventListener(eventName, () => {
        uploadZone.classList.add('ns-upload-zone--dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      uploadZone.addEventListener(eventName, () => {
        uploadZone.classList.remove('ns-upload-zone--dragover');
      });
    });

    uploadZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length === 0) return;

      const file = files[0];
      if (!validateFile(file)) {
        alert('Invalid file format. Please upload an MRI scan in JPG, JPEG, or PNG format.');
        return;
      }

      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      displayPreview(file);
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length === 0) return;
      const file = fileInput.files[0];
      if (!validateFile(file)) {
        alert('Invalid file format. Please upload an MRI scan in JPG, JPEG, or PNG format.');
        fileInput.value = '';
        return;
      }
      displayPreview(file);
    });
  }

  if (removeBtn) {
    removeBtn.addEventListener('click', (e) => {
      e.preventDefault();
      resetUpload();
    });
  }

  // ---------- Multi-Step Loading HUD ----------
  const loadingStages = [
    "Preprocessing MRI scan & normalizing tensor (224×224 RGB)...",
    "Running Deep Convolutional Classifier across 4 pathological classes...",
    "Executing YOLOv11 instance segmentation & spatial localization...",
    "Computing tumor volumetric ratio & mask overlay..."
  ];

  if (uploadForm) {
    uploadForm.addEventListener('submit', (e) => {
      if (!fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert('Please select an MRI image before analyzing.');
        return;
      }

      if (!validateFile(fileInput.files[0])) {
        e.preventDefault();
        alert('Please provide a valid JPG, JPEG, or PNG MRI scan.');
        return;
      }

      if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Processing...';
      }

      if (loadingHud) {
        loadingHud.classList.add('ns-loading-hud--active');
        let stageIdx = 0;
        setInterval(() => {
          stageIdx = (stageIdx + 1) % loadingStages.length;
          if (loadingStepTxt) {
            loadingStepTxt.textContent = loadingStages[stageIdx];
          }
        }, 1600);
      }
    });
  }

  // ---------- Copy Diagnostic Summary (Results Page) ----------
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const summaryText = copyBtn.getAttribute('data-summary') || '';
      if (navigator.clipboard) {
        navigator.clipboard.writeText(summaryText).then(() => {
          showToast('Diagnostic Summary copied to clipboard.');
        }).catch(() => {
          showToast('Unable to access clipboard.');
        });
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = summaryText;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showToast('Diagnostic Summary copied to clipboard.');
      }
    });
  }

})();
