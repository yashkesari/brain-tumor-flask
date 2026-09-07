/* ============================================================
   NEUROSCAN AI — Interactive Client
   ============================================================
   Handles TWO independent upload workflows:

   1. BRAIN MRI ANALYSIS
      - File: mri-file-input
      - Form: ns-mri-form
      - Endpoint: POST /predict

   2. MRI MEDICAL REPORT ANALYSIS
      - File: report-file-input
      - Form: ns-report-form
      - Endpoint: POST /analyze-report

   Each workflow has its own state, validation, and preview.
   No framework dependencies — pure Vanilla JS.
   ============================================================ */

(function () {
  'use strict';

  // ============================================================
  // SHARED UTILITIES
  // ============================================================

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    var k = 1024;
    var sizes = ['Bytes', 'KB', 'MB'];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function showToast(message) {
    var toast = document.getElementById('ns-toast');
    var toastMsg = document.getElementById('ns-toast-msg');
    if (!toast || !toastMsg) return;
    toastMsg.textContent = message;
    toast.classList.add('ns-toast--show');
    setTimeout(function () {
      toast.classList.remove('ns-toast--show');
    }, 3200);
  }

  function showLoadingHud(messages) {
    var hud = document.getElementById('ns-loading-hud');
    var stepText = document.getElementById('ns-loading-step-text');
    if (!hud) return;

    hud.classList.add('ns-loading-hud--active');

    if (stepText && messages && messages.length > 0) {
      stepText.textContent = messages[0];
      var idx = 0;
      setInterval(function () {
        idx = (idx + 1) % messages.length;
        stepText.textContent = messages[idx];
      }, 1600);
    }
  }


  // ============================================================
  // MODULE A: BRAIN MRI ANALYSIS
  // ============================================================

  var mriForm       = document.getElementById('ns-mri-form');
  var mriInput      = document.getElementById('mri-file-input');
  var mriZone       = document.getElementById('ns-mri-upload-zone');
  var mriFileInfo   = document.getElementById('ns-mri-file-info');
  var mriFileName   = document.getElementById('ns-mri-filename');
  var mriFileSize   = document.getElementById('ns-mri-filesize');
  var mriRemove     = document.getElementById('ns-mri-remove');
  var mriPreview    = document.getElementById('ns-mri-preview');
  var mriPreviewImg = document.getElementById('ns-mri-preview-img');
  var mriSubmit     = document.getElementById('ns-mri-submit');

  var MRI_ALLOWED_EXTS = ['.jpg', '.jpeg', '.png', '.pdf'];

  function validateMriFile(file) {
    if (!file) return false;
    var ext = '.' + file.name.split('.').pop().toLowerCase();
    return MRI_ALLOWED_EXTS.indexOf(ext) !== -1;
  }

  function showMriFile(file) {
    if (!file) return;

    // Show file info
    if (mriFileName) mriFileName.textContent = file.name;
    if (mriFileSize) mriFileSize.textContent = formatBytes(file.size);
    if (mriFileInfo) mriFileInfo.classList.add('ns-module-file-info--active');

    // Hide upload zone
    if (mriZone) mriZone.style.display = 'none';

    // Show image preview for image files
    var ext = '.' + file.name.split('.').pop().toLowerCase();
    if (ext !== '.pdf' && mriPreview && mriPreviewImg) {
      var reader = new FileReader();
      reader.onload = function (e) {
        mriPreviewImg.src = e.target.result;
        mriPreview.classList.add('ns-module-preview--active');
      };
      reader.readAsDataURL(file);
    }

    // Enable submit
    if (mriSubmit) mriSubmit.disabled = false;
  }

  function resetMri() {
    if (mriInput) mriInput.value = '';
    if (mriFileName) mriFileName.textContent = '';
    if (mriFileSize) mriFileSize.textContent = '';
    if (mriFileInfo) mriFileInfo.classList.remove('ns-module-file-info--active');
    if (mriPreview) mriPreview.classList.remove('ns-module-preview--active');
    if (mriPreviewImg) mriPreviewImg.src = '';
    if (mriZone) mriZone.style.display = '';
    if (mriSubmit) mriSubmit.disabled = true;
  }

  // Drag & drop for MRI
  if (mriZone && mriInput) {
    ['dragenter', 'dragover'].forEach(function (evt) {
      mriZone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
        mriZone.classList.add('ns-module-upload-zone--dragover');
      });
    });

    ['dragleave', 'drop'].forEach(function (evt) {
      mriZone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
        mriZone.classList.remove('ns-module-upload-zone--dragover');
      });
    });

    mriZone.addEventListener('drop', function (e) {
      var files = e.dataTransfer.files;
      if (files.length === 0) return;
      var file = files[0];
      if (!validateMriFile(file)) {
        alert('Please upload a supported MRI file: JPG, JPEG, PNG, or PDF.');
        return;
      }
      var dt = new DataTransfer();
      dt.items.add(file);
      mriInput.files = dt.files;
      showMriFile(file);
    });

    mriInput.addEventListener('change', function () {
      if (mriInput.files.length === 0) return;
      var file = mriInput.files[0];
      if (!validateMriFile(file)) {
        alert('Please upload a supported MRI file: JPG, JPEG, PNG, or PDF.');
        mriInput.value = '';
        return;
      }
      showMriFile(file);
    });
  }

  if (mriRemove) {
    mriRemove.addEventListener('click', function (e) {
      e.preventDefault();
      resetMri();
    });
  }

  // MRI form submit
  if (mriForm) {
    mriForm.addEventListener('submit', function (e) {
      if (!mriInput || !mriInput.files || mriInput.files.length === 0) {
        e.preventDefault();
        alert('Please select an MRI image before analyzing.');
        return;
      }
      if (!validateMriFile(mriInput.files[0])) {
        e.preventDefault();
        alert('Please upload a supported MRI file: JPG, JPEG, PNG, or PDF.');
        return;
      }
      if (mriSubmit) {
        mriSubmit.disabled = true;
        mriSubmit.textContent = 'Processing...';
      }
      showLoadingHud([
        'Preprocessing MRI scan & normalizing tensor (224×224 RGB)...',
        'Running Deep Convolutional Classifier across 4 pathological classes...',
        'Executing YOLOv11 instance segmentation & spatial localization...',
        'Computing tumor volumetric ratio & mask overlay...'
      ]);
    });
  }


  // ============================================================
  // MODULE B: MRI MEDICAL REPORT ANALYSIS
  // ============================================================

  var reportForm     = document.getElementById('ns-report-form');
  var reportInput    = document.getElementById('report-file-input');
  var reportZone     = document.getElementById('ns-report-upload-zone');
  var reportFileInfo = document.getElementById('ns-report-file-info');
  var reportFileName = document.getElementById('ns-report-filename');
  var reportFileSize = document.getElementById('ns-report-filesize');
  var reportRemove   = document.getElementById('ns-report-remove');
  var reportSubmit   = document.getElementById('ns-report-submit');

  var REPORT_ALLOWED_EXTS = ['.pdf', '.jpg', '.jpeg', '.png'];

  function validateReportFile(file) {
    if (!file) return false;
    var ext = '.' + file.name.split('.').pop().toLowerCase();
    return REPORT_ALLOWED_EXTS.indexOf(ext) !== -1;
  }

  function showReportFile(file) {
    if (!file) return;

    if (reportFileName) reportFileName.textContent = file.name;
    if (reportFileSize) reportFileSize.textContent = formatBytes(file.size);
    if (reportFileInfo) reportFileInfo.classList.add('ns-module-file-info--active');

    if (reportZone) reportZone.style.display = 'none';

    if (reportSubmit) reportSubmit.disabled = false;
  }

  function resetReport() {
    if (reportInput) reportInput.value = '';
    if (reportFileName) reportFileName.textContent = '';
    if (reportFileSize) reportFileSize.textContent = '';
    if (reportFileInfo) reportFileInfo.classList.remove('ns-module-file-info--active');
    if (reportZone) reportZone.style.display = '';
    if (reportSubmit) reportSubmit.disabled = true;
  }

  // Drag & drop for Report
  if (reportZone && reportInput) {
    ['dragenter', 'dragover'].forEach(function (evt) {
      reportZone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
        reportZone.classList.add('ns-module-upload-zone--dragover');
      });
    });

    ['dragleave', 'drop'].forEach(function (evt) {
      reportZone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
        reportZone.classList.remove('ns-module-upload-zone--dragover');
      });
    });

    reportZone.addEventListener('drop', function (e) {
      var files = e.dataTransfer.files;
      if (files.length === 0) return;
      var file = files[0];
      if (!validateReportFile(file)) {
        alert('Please upload a supported medical report: PDF, JPG, JPEG, or PNG.');
        return;
      }
      var dt = new DataTransfer();
      dt.items.add(file);
      reportInput.files = dt.files;
      showReportFile(file);
    });

    reportInput.addEventListener('change', function () {
      if (reportInput.files.length === 0) return;
      var file = reportInput.files[0];
      if (!validateReportFile(file)) {
        alert('Please upload a supported medical report: PDF, JPG, JPEG, or PNG.');
        reportInput.value = '';
        return;
      }
      showReportFile(file);
    });
  }

  if (reportRemove) {
    reportRemove.addEventListener('click', function (e) {
      e.preventDefault();
      resetReport();
    });
  }

  // Report form submit
  if (reportForm) {
    reportForm.addEventListener('submit', function (e) {
      if (!reportInput || !reportInput.files || reportInput.files.length === 0) {
        e.preventDefault();
        alert('Please select a medical report before analyzing.');
        return;
      }
      if (!validateReportFile(reportInput.files[0])) {
        e.preventDefault();
        alert('Please upload a supported medical report: PDF, JPG, JPEG, or PNG.');
        return;
      }
      if (reportSubmit) {
        reportSubmit.disabled = true;
        reportSubmit.textContent = 'Processing...';
      }
      showLoadingHud([
        'Extracting text from document using OCR...',
        'Analyzing document structure and type...',
        'Running NLP extraction on report sections...',
        'Generating structured findings summary...'
      ]);
    });
  }


  // ============================================================
  // GLOBAL: Prevent default drag behavior on body
  // ============================================================

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(function (evt) {
    document.body.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
    });
  });


  // ============================================================
  // RESULTS PAGE: Copy Diagnostic Summary
  // ============================================================

  var copyBtn = document.getElementById('ns-copy-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var summaryText = copyBtn.getAttribute('data-summary') || '';
      if (navigator.clipboard) {
        navigator.clipboard.writeText(summaryText).then(function () {
          showToast('Diagnostic Summary copied to clipboard.');
        }).catch(function () {
          showToast('Unable to access clipboard.');
        });
      } else {
        var textarea = document.createElement('textarea');
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
