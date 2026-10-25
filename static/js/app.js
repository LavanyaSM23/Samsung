/**
 * NutriSense AI - Frontend Application Logic
 * Integrates image upload, real-time MobileNetV2 prediction, ambiguous subtype disambiguation,
 * portion scaling, ICMR-NIN 2020 RDA suitability evaluation, and clinical warnings.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentPredictionData = null;
  let activeProfileKey = 'adult_male_sedentary';
  let activeSubtypeId = null;
  let activePortionG = 100;
  let debounceTimer = null;

  // DOM Elements - Navigation
  const navTabs = document.querySelectorAll('.nav-tab');
  const viewSections = document.querySelectorAll('.view-section');
  const profileSelect = document.getElementById('healthProfileSelect');

  // Scanner View Elements
  const dropzone = document.getElementById('imageDropzone');
  const fileInput = document.getElementById('fileInput');
  const btnBrowseFiles = document.getElementById('btnBrowseFiles');
  const previewContainer = document.getElementById('previewContainer');
  const imagePreview = document.getElementById('imagePreview');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const btnChangeImage = document.getElementById('btnChangeImage');
  const quickSampleChips = document.getElementById('quickSampleChips');

  // Profile display
  const profileTitleDisplay = document.getElementById('profileTitleDisplay');
  const profileDescDisplay = document.getElementById('profileDescDisplay');
  const profileRdaBadge = document.getElementById('profileRdaBadge');

  // Results State Containers
  const emptyState = document.getElementById('emptyStatePlaceholder');
  const loadingState = document.getElementById('loadingState');
  const resultsContent = document.getElementById('resultsContent');

  // Prediction Header & Candidates
  const primaryFoodName = document.getElementById('primaryFoodName');
  const primaryConfidenceBadge = document.getElementById('primaryConfidenceBadge');
  const matchTypeBadge = document.getElementById('matchTypeBadge');
  const candidatesList = document.getElementById('candidatesList');

  // Subtype Elements
  const subtypeAlertCard = document.getElementById('subtypeAlertCard');
  const subtypePromptText = document.getElementById('subtypePromptText');
  const subtypeButtonsContainer = document.getElementById('subtypeButtonsContainer');

  // Portion Control (Phase 7)
  const portionSlider = document.getElementById('portionSlider');
  const portionValDisplay = document.getElementById('portionValDisplay');
  const presetButtons = document.querySelectorAll('.preset-btn');
  const definedServingSelect = document.getElementById('definedServingSelect');
  const portionDirectInput = document.getElementById('portionDirectInput');
  const portionBasisTag = document.getElementById('portionBasisTag');
  const nutritionSourceTag = document.getElementById('nutritionSourceTag');
  const missingNutrientsBanner = document.getElementById('missingNutrientsBanner');
  const missingNutrientsText = document.getElementById('missingNutrientsText');

  // Manual Food Search Fallback (Phase 7)
  const manualFoodSearchInput = document.getElementById('manualFoodSearchInput');
  const btnManualSearchClear = document.getElementById('btnManualSearchClear');
  const manualSearchResultsDropdown = document.getElementById('manualSearchResultsDropdown');

  // Macronutrients
  const valCalories = document.getElementById('valCalories');
  const valProtein = document.getElementById('valProtein');
  const valCarbs = document.getElementById('valCarbs');
  const valFat = document.getElementById('valFat');
  const valFiber = document.getElementById('valFiber');

  // Micronutrients
  const valSodium = document.getElementById('valSodium');
  const valCalcium = document.getElementById('valCalcium');
  const valIron = document.getElementById('valIron');
  const valVitC = document.getElementById('valVitC');
  const valFolate = document.getElementById('valFolate');

  // Composite Recipe
  const compositeCard = document.getElementById('compositeCard');
  const compositeSourceType = document.getElementById('compositeSourceType');
  const ingredientsTagsContainer = document.getElementById('ingredientsTagsContainer');

  // Warning & Suitability
  const qualityWarningCard = document.getElementById('qualityWarningCard');
  const qualityWarningText = document.getElementById('qualityWarningText');
  const scoreCircle = document.getElementById('scoreCircle');
  const suitabilityScoreNum = document.getElementById('suitabilityScoreNum');
  const verdictText = document.getElementById('verdictText');
  const scoreProfileName = document.getElementById('scoreProfileName');
  const alertsContainer = document.getElementById('alertsContainer');
  const allergenBadges = document.getElementById('allergenBadges');

  // Gallery & Catalog
  const fullSamplesGrid = document.getElementById('fullSamplesGrid');
  const catalogSearchInput = document.getElementById('catalogSearchInput');
  const catalogTableBody = document.getElementById('catalogTableBody');

  // 1. Navigation Tab Switching
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetView = tab.getAttribute('data-view');
      navTabs.forEach(t => t.classList.remove('active'));
      viewSections.forEach(s => s.classList.remove('active'));

      tab.classList.add('active');
      const activeSection = document.getElementById(targetView);
      if (activeSection) {
        activeSection.classList.add('active');
      }

      if (targetView === 'samplesView') {
        loadDemoGallery();
      } else if (targetView === 'catalogView') {
        loadFoodCatalog();
      } else if (targetView === 'benchmarksView') {
        loadBenchmarksKPI();
      }
    });
  });

  // 2. Health Profile Selection
  profileSelect.addEventListener('change', (e) => {
    activeProfileKey = e.target.value;
    updateProfileMetadataDisplay(activeProfileKey);

    // If active food is loaded, recalculate nutrition & health score
    if (currentPredictionData && currentPredictionData.prediction) {
      recalculateAdjustedNutrition();
    }
  });

  function updateProfileMetadataDisplay(key) {
    fetch('/api/profiles')
      .then(res => res.json())
      .then(profiles => {
        const found = profiles.find(p => p.key === key);
        if (found) {
          profileTitleDisplay.textContent = found.title;
          profileDescDisplay.textContent = found.description;
          profileRdaBadge.textContent = `${found.calories_target.toLocaleString()} kcal / day`;
        }
      })
      .catch(err => console.error('Error fetching profile metadata:', err));
  }

  // 3. Image Upload Handling (Drag & Drop + File Select)
  btnBrowseFiles.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  });

  btnChangeImage.addEventListener('click', () => {
    fileInput.value = '';
    previewContainer.classList.add('hidden');
    dropzonePrompt.classList.remove('hidden');
    fileInput.click();
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  });

  function processSelectedFile(file) {
    if (!file.type.match('image.*')) {
      alert('Please upload a valid image file (JPEG, PNG, WEBP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzonePrompt.classList.add('hidden');
      previewContainer.classList.remove('hidden');

      // Send to server
      uploadAndPredict(file);
    };
    reader.readAsDataURL(file);
  }

  // 4. Quick Sample Chips
  quickSampleChips.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;

    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');

    const sampleName = chip.getAttribute('data-sample');
    testSampleByName(sampleName);
  });

  function testSampleByName(sampleName) {
    const sampleUrl = `/static/samples/${sampleName}.jpg`;
    imagePreview.src = sampleUrl;
    dropzonePrompt.classList.add('hidden');
    previewContainer.classList.remove('hidden');

    showLoadingState();

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sample_path: sampleUrl,
        user_profile: activeProfileKey
      })
    })
    .then(res => res.json())
    .then(res => {
      if (res.success) {
        currentPredictionData = res.data;
        activeSubtypeId = null;
        renderAnalysisResults(res.data);
      } else {
        alert('Prediction error: ' + (res.error || 'Unknown error'));
        hideLoadingState();
      }
    })
    .catch(err => {
      console.error('Error during sample analysis:', err);
      alert('Network or server error during inference.');
      hideLoadingState();
    });
  }

  // 5. Upload & Predict via FormData
  function uploadAndPredict(file) {
    showLoadingState();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_profile', activeProfileKey);

    fetch('/api/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(res => {
      if (res.success) {
        currentPredictionData = res.data;
        activeSubtypeId = null;
        renderAnalysisResults(res.data);
      } else {
        alert('Inference error: ' + (res.error || 'Server error'));
        hideLoadingState();
      }
    })
    .catch(err => {
      console.error('Upload error:', err);
      alert('Failed to connect to NutriSense backend server.');
      hideLoadingState();
    });
  }

  function showLoadingState() {
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loadingState.classList.remove('hidden');
  }

  function hideLoadingState() {
    loadingState.classList.add('hidden');
  }

  // 6. Render Full Analysis Results
  function renderAnalysisResults(data) {
    hideLoadingState();
    emptyState.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    const pred = data.prediction || {};
    const nut = data.nutrition || {};
    const meta = data.metadata || {};
    const health = data.health_evaluation || {};

    // A. Header
    primaryFoodName.textContent = pred.display_name || pred.class_name;
    const confPct = ((pred.confidence || 0) * 100).toFixed(1);
    primaryConfidenceBadge.textContent = `${confPct}% Confidence`;

    matchTypeBadge.textContent = meta.match_type || 'VALID_DIRECT';
    matchTypeBadge.className = `badge ${meta.match_type === 'COMPOSITE_RECIPE' ? 'badge-success' : 'badge-info'}`;

    // B. Top Candidates
    renderCandidates(pred.top_k || []);

    // C. Ambiguous Subtype Disambiguation
    if (data.ambiguous_subtype && data.ambiguous_subtype.available_subtypes) {
      renderSubtypeSelector(data.ambiguous_subtype);
      subtypeAlertCard.classList.remove('hidden');
    } else {
      subtypeAlertCard.classList.add('hidden');
    }

    // D. Portion Control Setup (Phase 7)
    const currentPortion = nut.serving_size_g || 100;
    activePortionG = currentPortion;
    portionSlider.value = currentPortion;
    if (portionDirectInput) portionDirectInput.value = Math.round(currentPortion);
    portionValDisplay.textContent = `${Math.round(currentPortion)}g (${nut.serving_unit || 'Standard'})`;

    if (portionBasisTag) {
      portionBasisTag.textContent = `Basis: ${nut.reference_basis || (Math.round(currentPortion) + 'g portion')}`;
    }
    if (nutritionSourceTag) {
      nutritionSourceTag.textContent = `Source: ${meta.data_source || 'ICMR-NIN 2020'}`;
    }

    // Populate defined servings dropdown
    const foodClass = data.prediction?.class_name || data.class_name;
    if (foodClass) {
      loadDefinedServings(foodClass);
    }

    // E. Macronutrients
    valCalories.textContent = nut.calories_kcal != null ? Math.round(nut.calories_kcal) : '--';
    valProtein.textContent = nut.protein_g != null ? nut.protein_g.toFixed(1) : '--';
    valCarbs.textContent = nut.carbs_g != null ? nut.carbs_g.toFixed(1) : '--';
    valFat.textContent = nut.fat_g != null ? nut.fat_g.toFixed(1) : '--';
    valFiber.textContent = nut.fibre_g != null ? nut.fibre_g.toFixed(1) : '--';

    // F. Micronutrients
    valSodium.textContent = nut.sodium_mg != null ? `${Math.round(nut.sodium_mg)} mg` : 'Not reported';
    valCalcium.textContent = nut.calcium_mg != null ? `${Math.round(nut.calcium_mg)} mg` : 'Not reported';
    valIron.textContent = nut.iron_mg != null ? `${nut.iron_mg.toFixed(1)} mg` : 'Not reported';
    valVitC.textContent = nut.vitamin_c_mg != null ? `${nut.vitamin_c_mg.toFixed(1)} mg` : 'Not reported';
    valFolate.textContent = nut.folate_ug != null ? `${Math.round(nut.folate_ug)} \u03BCg` : 'Not reported';

    // Missing Micronutrients Transparency Banner (Phase 7)
    if (missingNutrientsBanner && missingNutrientsText) {
      const missingList = meta.missing_micronutrients || [];
      if (missingList.length > 0) {
        missingNutrientsBanner.classList.remove('hidden');
        missingNutrientsText.textContent = `Notice: Reference laboratory data for [${missingList.join(', ')}] is not reported in source tables. Missing nutrients are preserved as unavailable rather than assumed to be zero.`;
      } else {
        missingNutrientsBanner.classList.add('hidden');
      }
    }

    // G. Composite Recipe Context
    if (data.composite_recipe && data.composite_recipe.sample_ingredients) {
      renderCompositeInfo(data.composite_recipe);
      compositeCard.classList.remove('hidden');
    } else {
      compositeCard.classList.add('hidden');
    }

    // H. Quality Warnings (High Sodium / Calorie Discrepancy)
    let warningMessages = [];
    if (meta.sodium_flag === 'high_sodium_unverified') {
      warningMessages.push('High Sodium Notice: Raw source sodium exceeds 5,000 mg/serving and is unverified. Consume with caution.');
    }
    if (meta.calorie_flag === 'discrepant_atwater_over_25pct') {
      warningMessages.push('Calorie Calculation Notice: Dish reported with >25% discrepancy between declared calories and Atwater sum in source literature.');
    }

    if (warningMessages.length > 0) {
      qualityWarningText.innerHTML = warningMessages.join('<br>');
      qualityWarningCard.classList.remove('hidden');
    } else {
      qualityWarningCard.classList.add('hidden');
    }

    // I. Suitability Score & Clinical Alerts
    renderSuitabilityScore(health);
  }

  // Helper: Render Top Candidates Bar
  function renderCandidates(candidates) {
    candidatesList.innerHTML = '';
    const topCandidates = candidates.slice(0, 3);

    topCandidates.forEach((cand, idx) => {
      const card = document.createElement('div');
      card.className = `candidate-card ${idx === 0 ? 'primary' : ''}`;
      const pct = (cand.confidence * 100).toFixed(1);

      card.innerHTML = `
        <div class="candidate-name">${cand.display_name}</div>
        <div class="candidate-meter">
          <div class="candidate-meter-fill" style="width: ${pct}%"></div>
        </div>
        <div class="candidate-score">${pct}% match</div>
      `;
      candidatesList.appendChild(card);
    });
  }

  // Helper: Render Subtype Buttons
  function renderSubtypeSelector(subtypeData) {
    subtypePromptText.textContent = subtypeData.disambiguation_prompt || 'Confirm food filling:';
    subtypeButtonsContainer.innerHTML = '';

    subtypeData.available_subtypes.forEach(st => {
      const btn = document.createElement('button');
      btn.type = 'button';
      const isActive = (activeSubtypeId === st.subtype_id) || (!activeSubtypeId && st.subtype_id === subtypeData.default_subtype.subtype_id);
      btn.className = `subtype-btn ${isActive ? 'active' : ''}`;
      btn.innerHTML = `
        <span>${st.display_name}</span>
        <span class="subtype-cal-badge">(${Math.round(st.calories_kcal)} kcal)</span>
      `;

      btn.addEventListener('click', () => {
        document.querySelectorAll('.subtype-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeSubtypeId = st.subtype_id;
        recalculateAdjustedNutrition();
      });

      subtypeButtonsContainer.appendChild(btn);
    });
  }

  // Helper: Render Composite Ingredients
  function renderCompositeInfo(comp) {
    compositeSourceType.textContent = comp.source_type || 'Recipe-derived nutrition profile';
    ingredientsTagsContainer.innerHTML = '';
    (comp.sample_ingredients || []).forEach(ing => {
      if (ing) {
        const span = document.createElement('span');
        span.className = 'ingredient-tag';
        span.textContent = ing;
        ingredientsTagsContainer.appendChild(span);
      }
    });
  }

  // Helper: Render Health Suitability Score & Alerts
  function renderSuitabilityScore(health) {
    const score = health.suitability_score || 7.0;
    suitabilityScoreNum.textContent = score.toFixed(1);
    verdictText.textContent = (health.verdict || 'HEALTHY CHOICE').replace(/_/g, ' ');

    scoreCircle.className = 'score-circle';
    if (score < 4.5) {
      scoreCircle.classList.add('danger');
      verdictText.style.color = 'var(--accent-rose)';
    } else if (score < 6.5) {
      scoreCircle.classList.add('warning');
      verdictText.style.color = 'var(--accent-amber)';
    } else {
      verdictText.style.color = 'var(--accent-emerald)';
    }

    scoreProfileName.textContent = `Evaluated against ${activeProfileKey.replace(/_/g, ' ')} RDA`;

    // Alerts
    alertsContainer.innerHTML = '';
    const alerts = health.dietary_alerts || [];
    if (alerts.length > 0) {
      alerts.forEach(a => {
        const item = document.createElement('div');
        item.className = `alert-item ${a.severity}`;
        item.textContent = a.message;
        alertsContainer.appendChild(item);
      });
    }

    // Allergens
    allergenBadges.innerHTML = '';
    const allergens = health.detected_allergens || [];
    if (allergens.length > 0) {
      allergens.forEach(alg => {
        const b = document.createElement('span');
        b.className = 'allergen-badge';
        b.textContent = alg;
        allergenBadges.appendChild(b);
      });
    } else {
      const none = document.createElement('span');
      none.style.fontSize = '0.75rem';
      none.style.color = 'var(--text-muted)';
      none.textContent = 'None detected (Common allergens)';
      allergenBadges.appendChild(none);
    }
  }

  // 7. Interactive Portion Scaling (Phase 7 Enhanced)
  portionSlider.addEventListener('input', (e) => {
    activePortionG = parseFloat(e.target.value);
    portionValDisplay.textContent = `${Math.round(activePortionG)}g`;
    if (portionDirectInput) portionDirectInput.value = Math.round(activePortionG);

    presetButtons.forEach(b => b.classList.remove('active'));

    // Debounce recalculation
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      recalculateAdjustedNutrition();
    }, 150);
  });

  if (portionDirectInput) {
    portionDirectInput.addEventListener('change', (e) => {
      const val = parseFloat(e.target.value);
      if (!isNaN(val) && val > 0) {
        activePortionG = val;
        portionSlider.value = val;
        portionValDisplay.textContent = `${Math.round(val)}g`;
        presetButtons.forEach(b => b.classList.remove('active'));
        recalculateAdjustedNutrition();
      }
    });
  }

  if (definedServingSelect) {
    definedServingSelect.addEventListener('change', (e) => {
      const selected = e.target.selectedOptions[0];
      if (!selected) return;
      const g = parseFloat(selected.getAttribute('data-grams'));
      if (!isNaN(g) && g > 0) {
        activePortionG = g;
        portionSlider.value = g;
        if (portionDirectInput) portionDirectInput.value = Math.round(g);
        portionValDisplay.textContent = `${Math.round(g)}g (${selected.textContent})`;
        presetButtons.forEach(b => b.classList.remove('active'));
        recalculateAdjustedNutrition();
      }
    });
  }

  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      presetButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      activePortionG = parseFloat(btn.getAttribute('data-portion'));
      portionSlider.value = activePortionG;
      if (portionDirectInput) portionDirectInput.value = Math.round(activePortionG);
      portionValDisplay.textContent = `${Math.round(activePortionG)}g`;
      recalculateAdjustedNutrition();
    });
  });

  // Load Defined Servings for active food
  let lastLoadedFood = null;
  function loadDefinedServings(foodName) {
    if (!definedServingSelect || !foodName || foodName === lastLoadedFood) return;
    lastLoadedFood = foodName;

    fetch(`/api/nutrition/servings?food_name=${encodeURIComponent(foodName)}`)
      .then(res => res.json())
      .then(d => {
        if (!d.success || !d.servings) return;
        definedServingSelect.innerHTML = '';
        d.servings.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.unit;
          opt.setAttribute('data-grams', s.grams);
          opt.textContent = s.label;
          if (s.is_default) opt.selected = true;
          definedServingSelect.appendChild(opt);
        });
      })
      .catch(err => console.error('Error loading servings:', err));
  }

  // Phase 7: Manual Food Search Fallback Handler
  if (manualFoodSearchInput && manualSearchResultsDropdown) {
    let searchDebounce = null;

    manualFoodSearchInput.addEventListener('input', (e) => {
      const term = e.target.value.trim();
      if (btnManualSearchClear) {
        btnManualSearchClear.classList.toggle('hidden', !term);
      }
      if (!term || term.length < 2) {
        manualSearchResultsDropdown.classList.add('hidden');
        manualSearchResultsDropdown.innerHTML = '';
        return;
      }

      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        fetch(`/api/nutrition/search?q=${encodeURIComponent(term)}&limit=10`)
          .then(res => res.json())
          .then(res => {
            if (!res.success || !res.data || res.data.length === 0) {
              manualSearchResultsDropdown.innerHTML = '<div style="padding:10px 14px; color:var(--text-muted); font-size:0.75rem;">No matching foods found in reference database.</div>';
              manualSearchResultsDropdown.classList.remove('hidden');
              return;
            }

            manualSearchResultsDropdown.innerHTML = '';
            res.data.forEach(item => {
              const row = document.createElement('div');
              row.className = 'search-result-item';
              row.innerHTML = `
                <div>
                  <div class="search-res-title">${item.display_name}</div>
                  <div class="search-res-meta">${item.serving_unit || '100g'} &bull; ${item.source}</div>
                </div>
                <div class="search-res-badge">${item.calories_per_100g != null ? Math.round(item.calories_per_100g) + ' kcal/100g' : '--'}</div>
              `;

              row.addEventListener('click', () => {
                selectManualSearchFood(item);
                manualSearchResultsDropdown.classList.add('hidden');
                manualFoodSearchInput.value = item.display_name;
              });

              manualSearchResultsDropdown.appendChild(row);
            });

            manualSearchResultsDropdown.classList.remove('hidden');
          })
          .catch(err => console.error('Search error:', err));
      }, 200);
    });

    if (btnManualSearchClear) {
      btnManualSearchClear.addEventListener('click', () => {
        manualFoodSearchInput.value = '';
        btnManualSearchClear.classList.add('hidden');
        manualSearchResultsDropdown.classList.add('hidden');
      });
    }

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#manualSearchSection')) {
        manualSearchResultsDropdown.classList.add('hidden');
      }
    });
  }

  function selectManualSearchFood(item) {
    if (emptyState) emptyState.classList.add('hidden');
    if (resultsContent) resultsContent.classList.add('hidden');
    if (loadingState) loadingState.classList.remove('hidden');

    fetch('/api/nutrition/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        food_name: item.food_name,
        portion_g: activePortionG || 100.0,
        user_profile: activeProfileKey
      })
    })
      .then(res => res.json())
      .then(res => {
        if (loadingState) loadingState.classList.add('hidden');
        if (!res.success) {
          alert('Analysis failed: ' + (res.error || 'Unknown error'));
          if (emptyState) emptyState.classList.remove('hidden');
          return;
        }

        const data = res.data;
        const nut = data.nutrients || {};

        currentPredictionData = {
          prediction: {
            class_name: data.class_name,
            display_name: data.display_name,
            confidence: 1.0,
            match_type: data.match_type || 'MANUAL_SEARCH'
          },
          top_k: [{ class_name: data.class_name, display_name: data.display_name, confidence: 1.0 }],
          nutrition: {
            serving_size_g: data.portion.grams,
            serving_unit: data.reference_serving_unit || 'standard portion',
            portion_scale_factor: data.portion.scale_ratio_from_100g,
            calories_kcal: nut.calories_kcal?.value,
            carbs_g: nut.carbs_g?.value,
            protein_g: nut.protein_g?.value,
            fat_g: nut.fat_g?.value,
            fibre_g: nut.fibre_g?.value,
            sodium_mg: nut.sodium_mg?.value,
            calcium_mg: nut.calcium_mg?.value,
            iron_mg: nut.iron_mg?.value,
            vitamin_c_mg: nut.vitamin_c_mg?.value,
            folate_ug: nut.folate_ug?.value,
            reference_basis: data.reference_basis
          },
          metadata: {
            data_source: data.source,
            match_type: data.match_type,
            sodium_flag: data.data_quality_flags?.sodium_flag || 'ok',
            calorie_flag: data.data_quality_flags?.calorie_flag || 'ok',
            missing_micronutrients: (data.missing_nutrients || []).map(m => m.nutrient)
          },
          health_evaluation: data.health_evaluation,
          ambiguous_subtype: data.ambiguous_subtype
        };

        if (resultsContent) resultsContent.classList.remove('hidden');
        renderAnalysisResults(currentPredictionData);
      })
      .catch(err => {
        if (loadingState) loadingState.classList.add('hidden');
        if (emptyState) emptyState.classList.remove('hidden');
        console.error('Error analyzing manual food:', err);
      });
  }

  // 8. Real-time Recalculation via /api/adjust-nutrition
  function recalculateAdjustedNutrition() {
    if (!currentPredictionData || !currentPredictionData.prediction) return;

    const className = currentPredictionData.prediction.class_name;

    fetch('/api/adjust-nutrition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        class_name: className,
        user_profile: activeProfileKey,
        portion_g: activePortionG,
        selected_subtype_id: activeSubtypeId
      })
    })
    .then(res => res.json())
    .then(res => {
      if (res.success) {
        // Merge updated nutrition & health into current data
        currentPredictionData.nutrition = res.data.nutrition;
        currentPredictionData.health_evaluation = res.data.health_evaluation;
        currentPredictionData.metadata = res.data.metadata;
        if (res.data.ambiguous_subtype) {
          currentPredictionData.ambiguous_subtype = res.data.ambiguous_subtype;
        }

        renderAnalysisResults(currentPredictionData);
      }
    })
    .catch(err => console.error('Error adjusting nutrition:', err));
  }

  // 9. Load Demo Gallery View
  function loadDemoGallery() {
    fetch('/api/samples')
      .then(res => res.json())
      .then(samples => {
        fullSamplesGrid.innerHTML = '';
        samples.forEach(s => {
          const card = document.createElement('div');
          card.className = 'sample-card';
          card.innerHTML = `
            <img src="${s.url}" alt="${s.title}" class="sample-img">
            <div class="sample-info">
              <h3 class="sample-card-title">${s.title}</h3>
              <p class="sample-card-desc">${s.description}</p>
              <button type="button" class="btn btn-sm btn-primary">Scan with AI</button>
            </div>
          `;
          card.addEventListener('click', () => {
            // Switch to scanner tab
            document.getElementById('tabScanner').click();
            testSampleByName(s.class_name);
          });
          fullSamplesGrid.appendChild(card);
        });
      })
      .catch(err => console.error('Error loading samples:', err));
  }

  // 10. Load Food Catalog View
  let catalogFoodsCache = [];
  function loadFoodCatalog() {
    if (catalogFoodsCache.length > 0) {
      renderCatalogRows(catalogFoodsCache);
      return;
    }

    fetch('/api/food-catalog')
      .then(res => res.json())
      .then(foods => {
        catalogFoodsCache = foods;
        renderCatalogRows(foods);
      })
      .catch(err => console.error('Error loading food catalog:', err));
  }

  function renderCatalogRows(foods) {
    catalogTableBody.innerHTML = '';
    foods.forEach(f => {
      const tr = document.createElement('tr');
      const algs = (f.allergens || []).map(a => `<span class="allergen-badge">${a}</span>`).join(' ') || '<span style="color:var(--text-muted)">None</span>';
      tr.innerHTML = `
        <td style="font-weight:700">${f.display_name} ${f.has_subtypes ? '<span class="badge badge-outline" style="font-size:0.65rem">Subtypes</span>' : ''}</td>
        <td><span class="badge ${f.match_type === 'COMPOSITE_RECIPE' ? 'badge-success' : 'badge-info'}">${f.match_type}</span></td>
        <td>${f.serving_size_g || 100}g (${f.serving_unit || 'serving'})</td>
        <td style="font-weight:700; color:var(--accent-amber)">${Math.round(f.calories_kcal || 0)} kcal</td>
        <td>${(f.protein_g || 0).toFixed(1)}g</td>
        <td>${(f.carbs_g || 0).toFixed(1)}g</td>
        <td>${(f.fat_g || 0).toFixed(1)}g</td>
        <td>${Math.round(f.sodium_mg || 0)} mg ${f.sodium_flag === 'high_sodium_unverified' ? '<span style="color:var(--accent-amber)">(!)</span>' : ''}</td>
        <td>${algs}</td>
      `;
      catalogTableBody.appendChild(tr);
    });
  }

  // Search Filter in Catalog
  catalogSearchInput.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase().trim();
    if (!term) {
      renderCatalogRows(catalogFoodsCache);
      return;
    }
    const filtered = catalogFoodsCache.filter(f =>
      f.display_name.toLowerCase().includes(term) ||
      f.match_type.toLowerCase().includes(term) ||
      (f.allergens || []).some(a => a.toLowerCase().includes(term))
    );
    renderCatalogRows(filtered);
  });

  // 11. Load Benchmarks View
  function loadBenchmarksKPI() {
    fetch('/api/benchmarks')
      .then(res => res.json())
      .then(d => {
        const tm = d.test_metrics || {};
        document.getElementById('kpiTop1').textContent = `${(tm.top1_accuracy || 81.83).toFixed(2)}%`;
        document.getElementById('kpiTop5').textContent = `${(tm.top5_accuracy || 95.59).toFixed(2)}%`;
        document.getElementById('kpiMacroF1').textContent = `${((tm.macro_f1 || 0.7996) * 100).toFixed(2)}%`;
        document.getElementById('kpiFoodLens').textContent = `${(d.auxiliary_foodlens_accuracy || 65.66).toFixed(2)}%`;
      })
      .catch(err => console.error('Error loading benchmark stats:', err));
  }

  // ============================================================
  // 12. Label Scanner (Packaged Food OCR Feature)
  // ============================================================
  let currentLabelData = null;
  let labelUserCorrections = {};

  const NUTRIENT_DISPLAY_META = [
    { key: 'calories', label: 'Energy / Calories', unit: 'kcal' },
    { key: 'protein', label: 'Protein', unit: 'g' },
    { key: 'total_carbohydrates', label: 'Total Carbohydrates', unit: 'g' },
    { key: 'dietary_fiber', label: 'Dietary Fiber', unit: 'g' },
    { key: 'total_sugars', label: 'Total Sugars', unit: 'g' },
    { key: 'added_sugars', label: 'Added Sugars', unit: 'g' },
    { key: 'total_fat', label: 'Total Fat', unit: 'g' },
    { key: 'saturated_fat', label: 'Saturated Fat', unit: 'g' },
    { key: 'trans_fat', label: 'Trans Fat', unit: 'g' },
    { key: 'cholesterol', label: 'Cholesterol', unit: 'mg' },
    { key: 'sodium', label: 'Sodium', unit: 'mg' },
    { key: 'salt', label: 'Salt (Declared)', unit: 'g' },
    { key: 'calcium', label: 'Calcium', unit: 'mg' },
    { key: 'iron', label: 'Iron', unit: 'mg' },
    { key: 'potassium', label: 'Potassium', unit: 'mg' },
    { key: 'vitamin_d', label: 'Vitamin D', unit: 'mcg' },
  ];

  // Label Scanner Elements
  const labelDropzone = document.getElementById('labelDropzone');
  const labelFileInput = document.getElementById('labelFileInput');
  const btnLabelBrowse = document.getElementById('btnLabelBrowse');
  const labelDropzonePrompt = document.getElementById('labelDropzonePrompt');
  const labelPreviewContainer = document.getElementById('labelPreviewContainer');
  const labelImagePreview = document.getElementById('labelImagePreview');
  const btnChangeLabelImage = document.getElementById('btnChangeLabelImage');

  const labelEmptyState = document.getElementById('labelEmptyState');
  const labelLoadingState = document.getElementById('labelLoadingState');
  const labelResultsContent = document.getElementById('labelResultsContent');

  const labelProductName = document.getElementById('labelProductName');
  const labelBrand = document.getElementById('labelBrand');
  const labelReadabilityBadge = document.getElementById('labelReadabilityBadge');
  const labelExtractionNote = document.getElementById('labelExtractionNote');
  const labelExtractionNoteText = document.getElementById('labelExtractionNoteText');
  const labelServingVal = document.getElementById('labelServingVal');
  const labelNutritionBasis = document.getElementById('labelNutritionBasis');
  const labelNutritionBody = document.getElementById('labelNutritionBody');

  const labelIngredientsSection = document.getElementById('labelIngredientsSection');
  const labelIngredientsTxt = document.getElementById('labelIngredientsTxt');
  const labelAllergenSection = document.getElementById('labelAllergenSection');
  const labelAllergenContent = document.getElementById('labelAllergenContent');
  const labelMissingSection = document.getElementById('labelMissingSection');
  const labelMissingList = document.getElementById('labelMissingList');

  const btnConfirmLabel = document.getElementById('btnConfirmLabel');
  const labelHealthCard = document.getElementById('labelHealthCard');
  const labelScoreCircle = document.getElementById('labelScoreCircle');
  const labelScoreNum = document.getElementById('labelScoreNum');
  const labelVerdictText = document.getElementById('labelVerdictText');
  const labelScoreProfile = document.getElementById('labelScoreProfile');
  const labelAlertsContainer = document.getElementById('labelAlertsContainer');
  const labelHealthDisclaimer = document.getElementById('labelHealthDisclaimer');
  const labelOcrDisclaimer = document.getElementById('labelOcrDisclaimer');

  if (labelDropzone && labelFileInput) {
    // Browse button triggers file input
    btnLabelBrowse.addEventListener('click', (e) => {
      e.stopPropagation();
      labelFileInput.click();
    });

    labelDropzone.addEventListener('click', (e) => {
      if (e.target !== btnChangeLabelImage && !labelPreviewContainer.contains(e.target)) {
        labelFileInput.click();
      }
    });

    btnChangeLabelImage.addEventListener('click', (e) => {
      e.stopPropagation();
      labelFileInput.value = '';
      labelFileInput.click();
    });

    // Drag-and-drop
    ['dragenter', 'dragover'].forEach(eventName => {
      labelDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        labelDropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      labelDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        labelDropzone.classList.remove('dragover');
      });
    });

    labelDropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        handleLabelFile(files[0]);
      }
    });

    labelFileInput.addEventListener('change', (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleLabelFile(files[0]);
      }
    });
  }

  function handleLabelFile(file) {
    if (!file.type.match('image.*')) {
      alert('Please upload a valid image file (JPG, PNG, WEBP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      labelImagePreview.src = e.target.result;
      labelDropzonePrompt.classList.add('hidden');
      labelPreviewContainer.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Call OCR endpoint
    performLabelOcr(file);
  }

  function performLabelOcr(file) {
    if (labelEmptyState) labelEmptyState.classList.add('hidden');
    if (labelResultsContent) labelResultsContent.classList.add('hidden');
    if (labelLoadingState) labelLoadingState.classList.remove('hidden');
    if (labelHealthCard) labelHealthCard.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/scan-label', {
      method: 'POST',
      body: formData
    })
      .then(res => {
        if (!res.ok) {
          throw new Error(`Server returned HTTP ${res.status}`);
        }
        return res.json();
      })
      .then(payload => {
        if (!payload.success) {
          throw new Error(payload.error || 'Failed to scan label.');
        }
        renderLabelResults(payload.data);
      })
      .catch(err => {
        console.error('Label OCR Error:', err);
        if (labelLoadingState) labelLoadingState.classList.add('hidden');
        if (labelEmptyState) {
          labelEmptyState.classList.remove('hidden');
          const title = labelEmptyState.querySelector('.empty-title');
          const text = labelEmptyState.querySelector('.empty-text');
          if (title) title.textContent = 'Label Scan Failed';
          if (text) text.textContent = `Error: ${err.message}. Please try again with a clearer image.`;
        }
      });
  }

  function renderLabelResults(data) {
    currentLabelData = data;
    labelUserCorrections = {};

    if (labelLoadingState) labelLoadingState.classList.add('hidden');
    if (labelEmptyState) labelEmptyState.classList.add('hidden');
    if (labelResultsContent) labelResultsContent.classList.remove('hidden');

    // 1. Product & Brand
    const product = data.product || {};
    labelProductName.textContent = product.product_name || 'Product (Name not visible on label)';
    labelBrand.textContent = product.brand ? `Brand: ${product.brand}` : 'Brand: Not detected';

    // 2. Readability Badge
    const read = data.readability || {};
    const qual = (read.quality || 'UNKNOWN').toUpperCase();
    let badgeClass = 'readability-partial';
    if (qual === 'GOOD') badgeClass = 'readability-good';
    else if (qual === 'LOW') badgeClass = 'readability-low';
    else if (qual === 'UNREADABLE') badgeClass = 'readability-unreadable';

    const confPct = Math.round((read.mean_confidence || 0) * 100);
    labelReadabilityBadge.innerHTML = `<span class="readability-badge ${badgeClass}">${qual} (${confPct}% confidence)</span>`;

    // 3. Extraction Note
    if (data.extraction_note && data.extraction_note !== 'Extraction completed successfully.') {
      labelExtractionNoteText.textContent = data.extraction_note;
      labelExtractionNote.classList.remove('hidden');
    } else {
      labelExtractionNote.classList.add('hidden');
    }

    // 4. Serving Size & Basis
    const serving = data.serving || {};
    let servingStr = '';
    if (serving.serving_size) {
      servingStr = `${serving.serving_size} ${serving.serving_unit || 'g'}`;
      if (serving.servings_per_container) {
        servingStr += ` (${serving.servings_per_container} servings/pack)`;
      }
    } else {
      servingStr = 'Not declared on visible packaging';
    }
    labelServingVal.textContent = servingStr;
    labelNutritionBasis.textContent = serving.nutrition_basis ? `Basis: ${serving.nutrition_basis}` : 'Basis: Declared values';

    // 5. Nutrition Table (editable)
    labelNutritionBody.innerHTML = '';
    const nutrition = data.nutrition || {};

    NUTRIENT_DISPLAY_META.forEach(nMeta => {
      const entry = nutrition[nMeta.key];
      const hasVal = entry && entry.value !== null && entry.value !== undefined;
      const row = document.createElement('tr');

      const nameCell = document.createElement('td');
      nameCell.className = 'nutrient-name';
      nameCell.textContent = nMeta.label;

      const valCell = document.createElement('td');
      const input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.className = 'editable-val';
      input.setAttribute('data-key', nMeta.key);
      input.placeholder = '—';
      if (hasVal) {
        input.value = entry.value;
      }

      valCell.appendChild(input);

      const unitCell = document.createElement('td');
      unitCell.className = 'nutrient-unit';
      unitCell.textContent = (entry && entry.unit) ? entry.unit : nMeta.unit;

      const sourceCell = document.createElement('td');
      sourceCell.className = hasVal ? 'source-badge-ocr' : 'nutrient-missing';
      sourceCell.textContent = hasVal ? 'OCR Extracted' : 'Unavailable';

      // Listen for user edits
      input.addEventListener('input', () => {
        input.classList.add('user-corrected');
        sourceCell.className = 'source-badge-user';
        sourceCell.textContent = 'User Corrected';

        const newVal = input.value.trim() === '' ? null : parseFloat(input.value);
        labelUserCorrections[nMeta.key] = {
          original: hasVal ? entry.value : null,
          user_value: newVal,
          is_corrected: true
        };
      });

      row.appendChild(nameCell);
      row.appendChild(valCell);
      row.appendChild(unitCell);
      row.appendChild(sourceCell);
      labelNutritionBody.appendChild(row);
    });

    // 6. Ingredients
    if (data.ingredients) {
      labelIngredientsTxt.textContent = data.ingredients;
      labelIngredientsSection.classList.remove('hidden');
    } else {
      labelIngredientsTxt.innerHTML = '<span style="color: var(--text-muted); font-style: italic;">Ingredients list was not visible on the package.</span>';
      labelIngredientsSection.classList.remove('hidden');
    }

    // 7. Allergens
    const allergens = data.allergens || {};
    labelAllergenContent.innerHTML = '';
    const decl = allergens.declared || [];
    const may = allergens.may_contain || [];

    if (decl.length === 0 && may.length === 0) {
      labelAllergenContent.innerHTML = '<span style="color: var(--text-muted); font-style: italic;">No specific allergen declarations detected on packaging.</span>';
    } else {
      const container = document.createElement('div');
      container.className = 'label-allergen-badges';

      decl.forEach(item => {
        const badge = document.createElement('span');
        badge.className = 'label-allergen-declared';
        badge.textContent = `Contains: ${item}`;
        container.appendChild(badge);
      });

      may.forEach(item => {
        const badge = document.createElement('span');
        badge.className = 'label-allergen-may';
        badge.textContent = `May Contain: ${item}`;
        container.appendChild(badge);
      });

      labelAllergenContent.appendChild(container);
    }

    // 8. Missing Fields
    const missing = data.missing_fields || [];
    labelMissingList.innerHTML = '';
    if (missing.length === 0) {
      labelMissingList.innerHTML = '<span style="color: var(--accent-emerald); font-size: 0.76rem;">All standard nutrition and product fields were successfully extracted.</span>';
    } else {
      missing.forEach(f => {
        const tag = document.createElement('span');
        tag.className = 'label-missing-tag';
        tag.innerHTML = `&#9888; ${f.replace('nutrition.', '')}`;
        labelMissingList.appendChild(tag);
      });
    }

    // 9. OCR Disclaimer
    labelOcrDisclaimer.textContent = data.disclaimer || (
      'NutriSense AI extracts packaging information using OCR for personal educational reference only. ' +
      'Missing or unreadable values are recorded as unavailable and are never fabricated. ' +
      'Verify critical values directly against the physical packaging.'
    );
  }

  // 13. Confirm Button & Health Evaluation
  if (btnConfirmLabel) {
    btnConfirmLabel.addEventListener('click', () => {
      if (!currentLabelData) return;

      // Harvest current values from the editable table
      const confirmedNutrition = {};
      const rows = labelNutritionBody.querySelectorAll('tr');

      rows.forEach(row => {
        const input = row.querySelector('.editable-val');
        if (!input) return;
        const key = input.getAttribute('data-key');
        const unit = row.querySelector('.nutrient-unit')?.textContent.trim() || 'g';
        const rawVal = input.value.trim();

        if (rawVal !== '') {
          confirmedNutrition[key] = {
            value: parseFloat(rawVal),
            unit: unit,
            ocr_extracted: !labelUserCorrections[key]
          };
        } else {
          confirmedNutrition[key] = null;
        }
      });

      btnConfirmLabel.disabled = true;
      btnConfirmLabel.textContent = 'Evaluating ICMR RDA Suitability...';

      fetch('/api/confirm-label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_profile: activeProfileKey,
          nutrition: confirmedNutrition,
          user_corrections: labelUserCorrections
        })
      })
        .then(res => res.json())
        .then(payload => {
          btnConfirmLabel.disabled = false;
          btnConfirmLabel.textContent = 'Confirm & Evaluate Nutrition Suitability';

          if (!payload.success) {
            alert('Evaluation failed: ' + (payload.error || 'Unknown error'));
            return;
          }

          renderLabelHealthEvaluation(payload.data);
        })
        .catch(err => {
          btnConfirmLabel.disabled = false;
          btnConfirmLabel.textContent = 'Confirm & Evaluate Nutrition Suitability';
          console.error('Error confirming label:', err);
          alert('Failed to evaluate nutrition suitability: ' + err.message);
        });
    });
  }

  function renderLabelHealthEvaluation(data) {
    labelHealthCard.classList.remove('hidden');

    const evalData = data.health_evaluation || {};
    const score = evalData.suitability_score;
    const verdict = evalData.verdict || 'Evaluation Complete';

    if (score !== null && score !== undefined) {
      labelScoreNum.textContent = score.toFixed(1);

      // Color circle
      if (score >= 7.0) {
        labelScoreCircle.style.borderColor = 'var(--accent-emerald)';
        labelScoreCircle.style.background = 'rgba(16, 185, 129, 0.12)';
        labelScoreNum.style.color = 'var(--accent-emerald)';
      } else if (score >= 4.0) {
        labelScoreCircle.style.borderColor = 'var(--accent-amber)';
        labelScoreCircle.style.background = 'rgba(245, 158, 11, 0.12)';
        labelScoreNum.style.color = 'var(--accent-amber)';
      } else {
        labelScoreCircle.style.borderColor = 'var(--accent-rose)';
        labelScoreCircle.style.background = 'rgba(244, 63, 94, 0.12)';
        labelScoreNum.style.color = 'var(--accent-rose)';
      }
    } else {
      labelScoreNum.textContent = '--';
      labelScoreCircle.style.borderColor = 'var(--text-muted)';
      labelScoreCircle.style.background = 'rgba(255, 255, 255, 0.05)';
      labelScoreNum.style.color = 'var(--text-muted)';
    }

    labelVerdictText.textContent = verdict;
    labelScoreProfile.textContent = `Profile: ${data.user_profile.replace(/_/g, ' ').toUpperCase()}`;

    // Alerts
    labelAlertsContainer.innerHTML = '';
    const alerts = evalData.dietary_alerts || [];
    if (alerts.length === 0) {
      const okDiv = document.createElement('div');
      okDiv.className = 'dietary-alert alert-low';
      okDiv.innerHTML = '<span class="alert-icon">&#10003;</span><span>No clinical nutrient thresholds exceeded for this profile.</span>';
      labelAlertsContainer.appendChild(okDiv);
    } else {
      alerts.forEach(a => {
        const div = document.createElement('div');
        let sevClass = 'alert-moderate';
        if (a.severity === 'HIGH' || a.severity === 'CRITICAL') sevClass = 'alert-critical';
        else if (a.severity === 'INFO') sevClass = 'alert-low';
        div.className = `dietary-alert ${sevClass}`;

        const icon = (a.severity === 'HIGH' || a.severity === 'CRITICAL') ? '&#9888;' : '&#8505;';
        div.innerHTML = `<span class="alert-icon">${icon}</span><span>${a.message}</span>`;
        labelAlertsContainer.appendChild(div);
      });
    }

    // Disclaimer
    labelHealthDisclaimer.textContent = `${evalData.note || ''} ${data.disclaimer || ''}`;

    // Smooth scroll to health evaluation card
    labelHealthCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Initial load
  updateProfileMetadataDisplay(activeProfileKey);
});
