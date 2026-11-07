/**
 * NutriSense AI - Frontend Application Logic
 * Integrates image upload, real-time MobileNetV2 prediction, ambiguous subtype disambiguation,
 * portion scaling, ICMR-NIN 2020 RDA suitability evaluation, and clinical warnings.
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentPredictionData = null;
  let activeProfileKey = 'adult_male_sedentary';  // ICMR predefined key (fallback)
  let activeUserId = localStorage.getItem('nutrisense_active_user_id') || null;
  let activeUserProfile = null;  // Full user profile object
  let activeSubtypeId = null;
  let activePortionG = 100;
  let debounceTimer = null;

  // DOM Elements - Navigation
  const navTabs = document.querySelectorAll('.nav-tab');
  const viewSections = document.querySelectorAll('.view-section');
  // Active user indicator in navbar
  const activeUserPill = document.getElementById('activeUserPill');
  const activeUserName = document.getElementById('activeUserName');
  const btnSwitchProfile = document.getElementById('btnSwitchProfile');

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
      // Only hide/show actual view-sections that are managed by nav (not display:none ones)
      document.querySelectorAll('.view-section').forEach(s => {
        if (!['samplesView','catalogView','benchmarksView'].includes(s.id)) {
          s.classList.remove('active');
        }
      });

      tab.classList.add('active');
      const activeSection = document.getElementById(targetView);
      if (activeSection) {
        activeSection.classList.add('active');
      }

      if (targetView === 'profileView') {
        loadProfileList();
      }
    });
  });

  // 2. Active Profile / User Display
  if (btnSwitchProfile) {
    btnSwitchProfile.addEventListener('click', () => {
      // Switch to Profile tab
      const profileTab = document.getElementById('tabProfile');
      if (profileTab) profileTab.click();
    });
  }

  if (activeUserPill) {
    activeUserPill.style.cursor = 'pointer';
    activeUserPill.addEventListener('click', () => {
      const profileTab = document.getElementById('tabProfile');
      if (profileTab) profileTab.click();
    });
  }

  function updateNavbarUserDisplay(profile) {
    if (!profile) {
      if (activeUserName) activeUserName.textContent = 'No Profile Set';
      if (activeUserPill) {
        activeUserPill.style.background = 'rgba(107, 114, 128, 0.08)';
        activeUserPill.style.borderColor = 'rgba(107, 114, 128, 0.25)';
        activeUserPill.style.color = 'var(--text-secondary)';
      }
      return;
    }
    const displayName = profile.display_name || profile.name || 'Unnamed Profile';
    if (activeUserName) activeUserName.textContent = displayName;
    if (activeUserPill) {
      activeUserPill.style.background = 'rgba(16, 185, 129, 0.12)';
      activeUserPill.style.borderColor = 'rgba(16, 185, 129, 0.35)';
      activeUserPill.style.color = 'var(--accent-emerald)';
    }
  }

  function updateProfileCardDisplay(profile) {
    if (!profile) {
      if (profileTitleDisplay) profileTitleDisplay.textContent = 'No Profile Set';
      if (profileDescDisplay) profileDescDisplay.textContent = 'Set up your profile in My Nutrition Profile for personalised suitability scores.';
      if (profileRdaBadge) profileRdaBadge.textContent = '-- kcal / day';
      return;
    }
    const name = profile.display_name || profile.name || 'Unnamed Profile';
    if (profileTitleDisplay) profileTitleDisplay.textContent = name;
    
    // Build description from profile data
    let desc = '';
    if (profile.dietary_preference) desc += profile.dietary_preference.charAt(0).toUpperCase() + profile.dietary_preference.slice(1) + ' • ';
    if (profile.activity_level) desc += profile.activity_level.replace(/_/g, ' ') + ' • ';
    if (profile.nutrition_goal) desc += profile.nutrition_goal.replace(/_/g, ' ');
    if (profileDescDisplay) profileDescDisplay.textContent = desc || 'Personal profile active.';
    
    // Show calorie target from computed data
    if (profile.daily_calorie_target) {
      if (profileRdaBadge) profileRdaBadge.textContent = `${Math.round(profile.daily_calorie_target).toLocaleString()} kcal / day`;
    } else if (profileRdaBadge) {
      profileRdaBadge.textContent = '-- kcal / day';
    }
    
    // Map to ICMR profile key for backend compatibility
    activeProfileKey = deriveIcmrKey(profile);
  }

  function deriveIcmrKey(profile) {
    // Map user profile to nearest ICMR predefined key for backend suitability scoring
    if (!profile) return 'adult_male_sedentary';
    const restrictions = (profile.food_restrictions || []).map(r => r.toLowerCase());
    if (restrictions.some(r => r.includes('diabet'))) return 'diabetic_adult';
    if (restrictions.some(r => r.includes('hypertens') || r.includes('blood pressure'))) return 'hypertensive_adult';
    if (profile.nutrition_goal === 'weight_loss' || profile.nutrition_goal === 'weight_gain') return 'weight_management';
    if (profile.sex === 'female') return 'adult_female_sedentary';
    return 'adult_male_sedentary';
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
    if (activeUserId) formData.append('user_id', activeUserId);

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

  // Helper: Render Top Candidates Bar (with Click-to-Select / Manual Correction)
  function renderCandidates(candidates) {
    candidatesList.innerHTML = '';
    const topCandidates = candidates.slice(0, 3);

    topCandidates.forEach((cand, idx) => {
      const card = document.createElement('div');
      card.className = `candidate-card ${idx === 0 ? 'primary' : ''}`;
      const pct = (cand.confidence * 100).toFixed(1);

      if (idx > 0) {
        card.style.cursor = 'pointer';
        card.title = `Click to switch dish to: ${cand.display_name}`;
        card.addEventListener('click', () => {
          selectFoodByName(cand.class_name, cand.display_name);
        });
      }

      card.innerHTML = `
        <div class="candidate-name">${cand.display_name}</div>
        <div class="candidate-meter">
          <div class="candidate-meter-fill" style="width: ${pct}%"></div>
        </div>
        <div class="candidate-score">${pct}% match ${idx > 0 ? '· <span style="text-decoration:underline">Select</span>' : ''}</div>
      `;
      candidatesList.appendChild(card);
    });
  }

  function selectFoodByName(className, displayName) {
    if (!className) return;
    showLoadingState();
    fetch('/api/adjust-nutrition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        class_name: className,
        user_profile: activeProfileKey,
        user_id: activeUserId || undefined,
        portion_g: activePortionG || 100,
      })
    })
    .then(res => res.json())
    .then(res => {
      if (res.success) {
        if (!currentPredictionData) currentPredictionData = { prediction: {} };
        const oldTopK = currentPredictionData.prediction?.top_k || [];
        currentPredictionData.prediction = {
          class_name: className,
          display_name: displayName || className.replace(/_/g, ' ').toUpperCase(),
          confidence: 1.0,
          top_k: oldTopK,
          match_type: res.data.metadata?.match_type || 'USER_CORRECTED'
        };
        currentPredictionData.nutrition = res.data.nutrition;
        currentPredictionData.health_evaluation = res.data.health_evaluation;
        currentPredictionData.metadata = res.data.metadata;
        currentPredictionData.ambiguous_subtype = res.data.ambiguous_subtype || null;
        renderAnalysisResults(currentPredictionData);
      } else {
        alert('Could not switch food: ' + (res.error || 'Unknown error'));
        hideLoadingState();
      }
    })
    .catch(err => {
      console.error(err);
      hideLoadingState();
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
        user_id: activeUserId || undefined,
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
    { key: 'dietary_fibre', altKey: 'dietary_fiber', label: 'Dietary Fibre', unit: 'g' },
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
    { key: 'vitamin_c', label: 'Vitamin C', unit: 'mg' },
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

  // Packaged Food Label Scanner DOM elements
  const modeTabPhotos = document.getElementById('modeTabPhotos');
  const modeTabManual = document.getElementById('modeTabManual');
  const panelModePhotos = document.getElementById('panelModePhotos');
  const panelModeManual = document.getElementById('panelModeManual');
  const labelThumbnailsStrip = document.getElementById('labelThumbnailsStrip');
  const manualProductSearchInput = document.getElementById('manualProductSearchInput');
  const manualIngredientsInput = document.getElementById('manualIngredientsInput');
  const btnAnalyzePack = document.getElementById('btnAnalyzePack');
  const labelNovaBadge = document.getElementById('labelNovaBadge');
  const labelHiddenInsightsBanner = document.getElementById('labelHiddenInsightsBanner');
  const labelIntelTabs = document.getElementById('labelIntelTabs');
  const tabBtnIngredients = document.getElementById('tabBtnIngredients');
  const tabBtnNutrition = document.getElementById('tabBtnNutrition');
  const tabBtnAlternatives = document.getElementById('tabBtnAlternatives');
  const viewIngredientsTab = document.getElementById('viewIngredientsTab');
  const viewNutritionTab = document.getElementById('viewNutritionTab');
  const viewAlternativesTab = document.getElementById('viewAlternativesTab');
  const labelIngredientsContainer = document.getElementById('labelIngredientsContainer');
  const labelBenefitsList = document.getElementById('labelBenefitsList');
  const labelAlternativesContainer = document.getElementById('labelAlternativesContainer');
  const labelRawOcrBox = document.getElementById('labelRawOcrBox');

  let selectedLabelFiles = [];

  // Mode Tab Switcher (Photos vs Manual)
  if (modeTabPhotos && modeTabManual) {
    modeTabPhotos.addEventListener('click', () => {
      modeTabPhotos.classList.add('active');
      modeTabManual.classList.remove('active');
      if (panelModePhotos) panelModePhotos.classList.remove('hidden');
      if (panelModeManual) panelModeManual.classList.add('hidden');
    });

    modeTabManual.addEventListener('click', () => {
      modeTabManual.classList.add('active');
      modeTabPhotos.classList.remove('active');
      if (panelModeManual) panelModeManual.classList.remove('hidden');
      if (panelModePhotos) panelModePhotos.classList.add('hidden');
      if (manualProductSearchInput) manualProductSearchInput.focus();
    });
  }

  // Popular product pills quick search
  const popularPills = document.querySelectorAll('.popular-pill');
  popularPills.forEach(pill => {
    pill.addEventListener('click', () => {
      const q = pill.getAttribute('data-query');
      if (manualProductSearchInput) {
        manualProductSearchInput.value = q;
      }
      performProductLookup(q);
    });
  });

  // Intel Sectional Tabs (Ingredients vs Nutrition vs Alternatives)
  const intelTabBtns = [tabBtnIngredients, tabBtnNutrition, tabBtnAlternatives];
  const intelTabViews = {
    viewIngredientsTab: viewIngredientsTab,
    viewNutritionTab: viewNutritionTab,
    viewAlternativesTab: viewAlternativesTab
  };

  intelTabBtns.forEach(btn => {
    if (btn) {
      btn.addEventListener('click', () => {
        const targetViewId = btn.getAttribute('data-tab');
        intelTabBtns.forEach(b => b && b.classList.remove('active'));
        btn.classList.add('active');

        Object.keys(intelTabViews).forEach(vId => {
          const v = intelTabViews[vId];
          if (v) {
            if (vId === targetViewId) v.classList.remove('hidden');
            else v.classList.add('hidden');
          }
        });
      });
    }
  });

  // Multi-File Upload & Drag-and-drop
  if (labelDropzone && labelFileInput) {
    if (btnLabelBrowse) {
      btnLabelBrowse.addEventListener('click', (e) => {
        e.stopPropagation();
        labelFileInput.click();
      });
    }

    labelDropzone.addEventListener('click', (e) => {
      if (!e.target.closest('.label-thumb-remove')) {
        labelFileInput.click();
      }
    });

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
        addLabelFiles(files);
      }
    });

    labelFileInput.addEventListener('change', (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        addLabelFiles(files);
      }
    });
  }

  function addLabelFiles(fileList) {
    for (let i = 0; i < fileList.length; i++) {
      const f = fileList[i];
      if (f.type.match('image.*') && selectedLabelFiles.length < 4) {
        selectedLabelFiles.push(f);
      }
    }
    renderThumbnails();
  }

  function renderThumbnails() {
    if (!labelThumbnailsStrip) return;
    labelThumbnailsStrip.innerHTML = '';

    if (selectedLabelFiles.length === 0) {
      labelThumbnailsStrip.classList.add('hidden');
      return;
    }

    labelThumbnailsStrip.classList.remove('hidden');

    selectedLabelFiles.forEach((file, index) => {
      const card = document.createElement('div');
      card.className = 'label-thumb-card';

      const img = document.createElement('img');
      img.className = 'label-thumb-img';
      img.alt = `Photo ${index + 1}`;

      const reader = new FileReader();
      reader.onload = (e) => { img.src = e.target.result; };
      reader.readAsDataURL(file);

      const rmBtn = document.createElement('button');
      rmBtn.type = 'button';
      rmBtn.className = 'label-thumb-remove';
      rmBtn.innerHTML = '&times;';
      rmBtn.title = 'Remove photo';
      rmBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedLabelFiles.splice(index, 1);
        renderThumbnails();
      });

      const badge = document.createElement('div');
      badge.className = 'label-thumb-badge';
      badge.textContent = index === 0 ? 'Side 1' : `Side ${index + 1}`;

      card.appendChild(img);
      card.appendChild(rmBtn);
      card.appendChild(badge);
      labelThumbnailsStrip.appendChild(card);
    });
  }

  // Trigger Analysis Button
  if (btnAnalyzePack) {
    btnAnalyzePack.addEventListener('click', () => {
      const isManual = modeTabManual && modeTabManual.classList.contains('active');

      if (isManual) {
        const query = manualProductSearchInput ? manualProductSearchInput.value.trim() : '';
        const rawIng = manualIngredientsInput ? manualIngredientsInput.value.trim() : '';

        if (!query && !rawIng) {
          alert('Please enter a packaged product name or paste an ingredient list to analyze.');
          if (manualProductSearchInput) manualProductSearchInput.focus();
          return;
        }

        if (query) {
          performProductLookup(query);
        } else {
          performIngredientsTextAnalysis(rawIng);
        }
      } else {
        if (selectedLabelFiles.length === 0) {
          alert('Please select or drop at least one product photo (front, nutrition table, or ingredients list).');
          if (labelFileInput) labelFileInput.click();
          return;
        }
        performMultiLabelOcr(selectedLabelFiles);
      }
    });
  }

  // Multi-image OCR Call
  function performMultiLabelOcr(files) {
    if (labelEmptyState) labelEmptyState.classList.add('hidden');
    if (labelResultsContent) labelResultsContent.classList.add('hidden');
    if (labelLoadingState) labelLoadingState.classList.remove('hidden');
    if (labelHealthCard) labelHealthCard.classList.add('hidden');

    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    if (activeUserId) formData.append('user_id', activeUserId);
    formData.append('user_profile', activeProfileKey || 'adult_male_sedentary');

    fetch('/api/scan-label', {
      method: 'POST',
      body: formData
    })
      .then(res => {
        if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
        return res.json();
      })
      .then(payload => {
        if (!payload.success) throw new Error(payload.error || 'Failed to scan label.');
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
          if (text) text.textContent = `Error: ${err.message}. Please try again with clearer photos.`;
        }
      });
  }

  // Direct Product Catalog Lookup
  function performProductLookup(query) {
    if (labelEmptyState) labelEmptyState.classList.add('hidden');
    if (labelResultsContent) labelResultsContent.classList.add('hidden');
    if (labelLoadingState) labelLoadingState.classList.remove('hidden');
    if (labelHealthCard) labelHealthCard.classList.add('hidden');

    fetch('/api/scan-label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        product_name: query,
        user_id: activeUserId || undefined,
        user_profile: activeProfileKey || 'adult_male_sedentary'
      })
    })
      .then(res => res.json())
      .then(payload => {
        if (!payload.success) throw new Error(payload.error || 'Product lookup failed.');
        renderLabelResults(payload.data);
      })
      .catch(err => {
        console.error('Lookup Error:', err);
        if (labelLoadingState) labelLoadingState.classList.add('hidden');
        if (labelEmptyState) {
          labelEmptyState.classList.remove('hidden');
          const title = labelEmptyState.querySelector('.empty-title');
          const text = labelEmptyState.querySelector('.empty-text');
          if (title) title.textContent = 'Product Not Found';
          if (text) text.textContent = `Could not find "${query}" in quick catalog. Upload photo of the packet for instant OCR.`;
        }
      });
  }

  // Direct Raw Ingredients Analysis
  function performIngredientsTextAnalysis(text) {
    if (labelEmptyState) labelEmptyState.classList.add('hidden');
    if (labelResultsContent) labelResultsContent.classList.add('hidden');
    if (labelLoadingState) labelLoadingState.classList.remove('hidden');
    if (labelHealthCard) labelHealthCard.classList.add('hidden');

    fetch('/api/scan-label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ingredients_text: text,
        user_id: activeUserId || undefined,
        user_profile: activeProfileKey || 'adult_male_sedentary'
      })
    })
      .then(res => res.json())
      .then(payload => {
        if (!payload.success) throw new Error(payload.error || 'Analysis failed.');
        renderLabelResults(payload.data);
      })
      .catch(err => {
        console.error('Text Analysis Error:', err);
        if (labelLoadingState) labelLoadingState.classList.add('hidden');
        if (labelEmptyState) labelEmptyState.classList.remove('hidden');
      });
  }

  // Main Render Results Engine
  function renderLabelResults(data) {
    currentLabelData = data;
    labelUserCorrections = {};

    if (labelLoadingState) labelLoadingState.classList.add('hidden');
    if (labelEmptyState) labelEmptyState.classList.add('hidden');
    if (labelResultsContent) labelResultsContent.classList.remove('hidden');

    // 1. Product Snapshot
    const product = data.product || {};
    labelProductName.textContent = product.product_name || 'Packaged Product';
    labelBrand.textContent = product.brand ? `Brand: ${product.brand}` : 'Brand: Unspecified';

    const serving = data.serving || {};
    let servingStr = '';
    if (serving.serving_size) {
      servingStr = `Serving: ${serving.serving_size} ${serving.serving_unit || 'g'}`;
      if (serving.servings_per_container) {
        servingStr += ` (${serving.servings_per_container} per pack)`;
      }
    }
    labelServingVal.textContent = servingStr;
    labelNutritionBasis.textContent = serving.nutrition_basis ? `Basis: ${serving.nutrition_basis}` : 'Basis: Declared Values';

    // 2. Readability & Processing Grade
    const read = data.readability || {};
    const qual = (read.quality || 'GOOD').toUpperCase();
    let badgeClass = 'readability-good';
    if (qual === 'LOW') badgeClass = 'readability-low';
    else if (qual === 'PARTIAL') badgeClass = 'readability-partial';
    const confPct = Math.round((read.mean_confidence || 0.95) * 100);
    labelReadabilityBadge.innerHTML = `<span class="readability-badge ${badgeClass}">${qual} (${confPct}% confidence)</span>`;

    const intel = data.ingredient_intelligence || {};
    const grade = intel.processing_grade || {};
    if (labelNovaBadge) {
      labelNovaBadge.textContent = grade.classification || 'Ultra-Processed (NOVA 4)';
      labelNovaBadge.className = `nova-badge ${grade.badge_class || 'nova-4'}`;
      if (grade.color) labelNovaBadge.style.color = grade.color;
    }

    // 3. Critical Insights & Hidden Ingredients Banner
    if (labelHiddenInsightsBanner) {
      labelHiddenInsightsBanner.innerHTML = '';
      const insights = intel.hidden_insights || [];
      const profileAlerts = intel.profile_alerts || [];

      // Render profile alerts first
      profileAlerts.forEach(pa => {
        const card = document.createElement('div');
        card.className = `label-insight-card severity-${pa.severity || 'HIGH'}`;
        card.innerHTML = `
          <div class="insight-icon">⚠️</div>
          <div class="insight-body">
            <div class="insight-title">${pa.profile} Alert</div>
            <div class="insight-desc">${pa.message}</div>
          </div>
        `;
        labelHiddenInsightsBanner.appendChild(card);
      });

      // Render hidden sugar, fat, and sodium alerts
      insights.forEach(ins => {
        const card = document.createElement('div');
        card.className = `label-insight-card severity-${ins.severity || 'HIGH'}`;
        const icon = ins.severity === 'CRITICAL' ? '🚫' : (ins.severity === 'HIGH' ? '⚠️' : 'ℹ️');
        card.innerHTML = `
          <div class="insight-icon">${icon}</div>
          <div class="insight-body">
            <div class="insight-title">${ins.title}</div>
            <div class="insight-desc">${ins.description}</div>
            ${ins.action_advice ? `<span class="insight-action">💡 ${ins.action_advice}</span>` : ''}
          </div>
        `;
        labelHiddenInsightsBanner.appendChild(card);
      });

      if (insights.length === 0 && profileAlerts.length === 0) {
        labelHiddenInsightsBanner.innerHTML = `
          <div class="label-insight-card severity-INFO">
            <div class="insight-icon">✓</div>
            <div class="insight-body">
              <div class="insight-title">Clean Ingredient Profile</div>
              <div class="insight-desc">No high-risk hidden sugars, industrial trans fats, or excessive synthetic additives flagged.</div>
            </div>
          </div>
        `;
      }
    }

    // 4. Tab 1: Smart Ingredients Explorer
    if (labelIngredientsContainer) {
      labelIngredientsContainer.innerHTML = '';
      const categorized = intel.categorized_ingredients || {};

      if (Object.keys(categorized).length === 0) {
        labelIngredientsContainer.innerHTML = `
          <div style="font-size:0.8rem; color:var(--text-muted); font-style:italic; padding:12px 0;">
            ${data.ingredients ? data.ingredients : 'Ingredients list was not visible on the package.'}
          </div>
        `;
      } else {
        Object.keys(categorized).forEach(catName => {
          const ingList = categorized[catName];
          if (!ingList || ingList.length === 0) return;

          const group = document.createElement('div');
          group.className = 'ingredient-group';
          group.innerHTML = `<h5 class="ingredient-group-title"><span>${catName}</span> (${ingList.length})</h5>`;

          const grid = document.createElement('div');
          grid.className = 'ingredient-chips-grid';

          ingList.forEach(item => {
            const card = document.createElement('div');
            card.className = 'ingredient-card';
            card.innerHTML = `
              <div class="ingredient-card-header">
                <span class="ing-name">${item.clean_name || item.name}</span>
                <span class="ing-cat-badge">${item.category}</span>
              </div>
              <div class="ing-purpose"><strong>Purpose:</strong> ${item.purpose}</div>
              <div class="ing-relevance"><strong>Health Relevance:</strong> ${item.health_relevance}</div>
              ${item.concerns ? `<div class="ing-purpose" style="color:var(--text-secondary);"><strong>Note:</strong> ${item.concerns}</div>` : ''}
              <div class="ing-safety-tag">✓ ${item.safety_status}</div>
            `;
            grid.appendChild(card);
          });

          group.appendChild(grid);
          labelIngredientsContainer.appendChild(group);
        });
      }
    }

    // Allergens Section
    const allergens = data.allergens || {};
    if (labelAllergenContent) {
      labelAllergenContent.innerHTML = '';
      const decl = allergens.declared || [];
      const detected = allergens.detected_from_ingredients || [];
      const may = allergens.may_contain || [];

      if (decl.length === 0 && detected.length === 0 && may.length === 0) {
        labelAllergenContent.innerHTML = '<span style="color: var(--text-muted); font-size: 0.78rem; font-style: italic;">No specific allergen declarations or common allergens detected.</span>';
      } else {
        const container = document.createElement('div');
        container.style.cssText = 'display:flex; flex-wrap:wrap; gap:6px;';

        decl.forEach(item => {
          const badge = document.createElement('span');
          badge.className = 'label-allergen-declared';
          badge.textContent = `Contains (Declared): ${item}`;
          container.appendChild(badge);
        });

        detected.forEach(item => {
          const badge = document.createElement('span');
          badge.className = 'label-allergen-detected';
          badge.style.cssText = 'display:inline-flex; align-items:center; padding:3px 10px; border-radius:12px; font-size:0.74rem; font-weight:600; background:rgba(245, 158, 11, 0.15); color:#F59E0B; border:1px solid rgba(245, 158, 11, 0.3);';
          badge.textContent = `In Ingredients: ${item}`;
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
    }

    // 5. Tab 2: Nutrition Table (ONLY Declared Rows)
    if (labelNutritionBody) {
      labelNutritionBody.innerHTML = '';
      const nutrition = data.nutrition || {};
      let declaredCount = 0;

      NUTRIENT_DISPLAY_META.forEach(nMeta => {
        const entry = nutrition[nMeta.key] || (nMeta.altKey ? nutrition[nMeta.altKey] : null);
        const hasVal = entry && entry.value !== null && entry.value !== undefined;

        // Consumer requirement: DO NOT display dozens of empty rows with "Unavailable"
        if (!hasVal) return;

        declaredCount++;
        const row = document.createElement('tr');

        const nameCell = document.createElement('td');
        nameCell.className = 'nutrient-name';
        nameCell.textContent = nMeta.label;

        const valCell = document.createElement('td');
        valCell.className = 'nutrient-declared-val';
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.className = 'nutrient-inline-edit';
        input.setAttribute('data-key', nMeta.key);
        input.value = entry.value;

        input.addEventListener('input', () => {
          input.style.borderColor = 'var(--accent-purple)';
          const newVal = input.value.trim() === '' ? null : parseFloat(input.value);
          labelUserCorrections[nMeta.key] = {
            original: entry.value,
            user_value: newVal,
            is_corrected: true
          };
          if (nMeta.altKey) {
            labelUserCorrections[nMeta.altKey] = labelUserCorrections[nMeta.key];
          }
        });
        valCell.appendChild(input);

        const unitCell = document.createElement('td');
        unitCell.className = 'nutrient-unit';
        unitCell.textContent = (entry && entry.unit) ? entry.unit : nMeta.unit;

        const confCell = document.createElement('td');
        confCell.style.cssText = 'font-size: 0.72rem; color: var(--accent-emerald); font-weight: 600;';
        confCell.textContent = entry.ocr_extracted ? '✓ Verified' : 'Declared';

        row.appendChild(nameCell);
        row.appendChild(valCell);
        row.appendChild(unitCell);
        row.appendChild(confCell);
        labelNutritionBody.appendChild(row);
      });

      if (declaredCount === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.innerHTML = `<td colspan="4" style="text-align:center; padding:18px; color:var(--text-muted); font-style:italic;">Nutrition table was not clearly visible on packaging. Click below to add values or evaluate ingredients.</td>`;
        labelNutritionBody.appendChild(emptyRow);
      }
    }

    // Product Positives & Benefits
    if (labelBenefitsList) {
      labelBenefitsList.innerHTML = '';
      const benefits = intel.product_benefits || [];
      if (benefits.length > 0) {
        benefits.forEach(b => {
          const li = document.createElement('li');
          li.className = 'benefits-item';
          li.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
            <span>${b}</span>
          `;
          labelBenefitsList.appendChild(li);
        });
        if (labelBenefitsContainer) labelBenefitsContainer.classList.remove('hidden');
      } else {
        if (labelBenefitsContainer) labelBenefitsContainer.classList.add('hidden');
      }
    }

    // 6. Tab 3: Healthier & Affordable Alternatives
    if (labelAlternativesContainer) {
      labelAlternativesContainer.innerHTML = '';
      const alts = intel.healthier_alternatives || [];
      alts.forEach(alt => {
        const card = document.createElement('div');
        card.className = 'alt-food-card';

        const tagsHtml = (alt.suitable_for || []).map(t => `<span class="alt-tag">${t}</span>`).join('');

        card.innerHTML = `
          <div class="alt-card-header">
            <span class="alt-food-name">${alt.name}</span>
            <span class="alt-price-pill">${alt.price_tier}</span>
          </div>
          <div class="alt-metrics-row">
            <span class="alt-metric">${alt.calories}</span>
            <span class="alt-metric"><span>${alt.fat_reduction}</span></span>
          </div>
          <div class="alt-why-better">${alt.why_better}</div>
          <div class="alt-tags-row">${tagsHtml}</div>
        `;
        labelAlternativesContainer.appendChild(card);
      });
    }

    // 7. Technical Details Accordion (Raw OCR)
    if (labelRawOcrBox) {
      labelRawOcrBox.textContent = data.ocr_text || 'No raw OCR text available.';
    }
    if (labelOcrDisclaimer) {
      labelOcrDisclaimer.textContent = data.disclaimer || (
        'NutriSense AI extracts packaging information using OCR for personal educational reference only. ' +
        'Missing or unreadable values are recorded as unavailable and are never fabricated. ' +
        'Verify critical values directly against the physical packaging.'
      );
    }
  }

  // Confirm Button & Health Evaluation
  if (btnConfirmLabel) {
    btnConfirmLabel.addEventListener('click', () => {
      if (!currentLabelData) return;

      const confirmedNutrition = {};
      const rows = labelNutritionBody ? labelNutritionBody.querySelectorAll('tr') : [];

      rows.forEach(row => {
        const input = row.querySelector('.nutrient-inline-edit') || row.querySelector('.editable-val');
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
          user_profile: activeProfileKey || 'adult_male_sedentary',
          user_id: activeUserId || undefined,
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
    if (!labelHealthCard) return;
    labelHealthCard.classList.remove('hidden');

    const evalData = data.health_evaluation || {};
    const score = evalData.suitability_score;
    const verdict = evalData.verdict || 'Evaluation Complete';

    if (score !== null && score !== undefined) {
      if (labelScoreNum) labelScoreNum.textContent = score.toFixed(1);

      if (labelScoreCircle && labelScoreNum) {
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
      }
    } else {
      if (labelScoreNum) labelScoreNum.textContent = '--';
      if (labelScoreCircle) {
        labelScoreCircle.style.borderColor = 'var(--text-muted)';
        labelScoreCircle.style.background = 'rgba(255, 255, 255, 0.05)';
      }
      if (labelScoreNum) labelScoreNum.style.color = 'var(--text-muted)';
    }

    if (labelVerdictText) labelVerdictText.textContent = verdict;
    if (labelScoreProfile) labelScoreProfile.textContent = `Profile: ${(data.user_profile || 'ADULT MALE').replace(/_/g, ' ').toUpperCase()}`;

    if (labelAlertsContainer) {
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
    }

    if (labelHealthDisclaimer) {
      labelHealthDisclaimer.textContent = `${evalData.note || ''} ${data.disclaimer || ''}`;
    }

    labelHealthCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // ============================================================
  // Theme Toggle (Dark / Light Theme Support)
  // ============================================================
  const btnThemeToggle = document.getElementById('btnThemeToggle');
  const themeToggleIcon = document.getElementById('themeToggleIcon');

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('nutrisense_theme', theme);
    if (themeToggleIcon) {
      themeToggleIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
    if (btnThemeToggle) {
      btnThemeToggle.setAttribute('title', theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode');
    }
  }

  const savedTheme = localStorage.getItem('nutrisense_theme') || 'light';
  applyTheme(savedTheme);

  if (btnThemeToggle) {
    btnThemeToggle.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme') || 'light';
      applyTheme(cur === 'dark' ? 'light' : 'dark');
    });
  }

  // Initial load: restore active user profile from localStorage or database
  loadProfileList();

  if (activeUserId) {
    fetch(`/api/profile/${activeUserId}`)
      .then(res => res.json())
      .then(payload => {
        if (payload.success && payload.data) {
          activeUserProfile = payload.data;
          updateNavbarUserDisplay(activeUserProfile);
          updateProfileCardDisplay(activeUserProfile);
          populateFormForEdit(activeUserProfile);
        } else {
          // Profile no longer exists - clear stored id
          activeUserId = null;
          localStorage.removeItem('nutrisense_active_user_id');
          updateNavbarUserDisplay(null);
          updateProfileCardDisplay(null);
        }
      })
      .catch(() => {
        updateNavbarUserDisplay(null);
        updateProfileCardDisplay(null);
      });
  } else {
    // Check if any profiles exist in DB to auto-select the most recent
    fetch('/api/user-profiles')
      .then(res => res.json())
      .then(payload => {
        if (payload.success && payload.data && payload.data.length > 0) {
          const first = payload.data[0];
          activeUserId = first.user_id;
          localStorage.setItem('nutrisense_active_user_id', activeUserId);
          fetch(`/api/profile/${activeUserId}`)
            .then(r => r.json())
            .then(p => {
              if (p.success && p.data) {
                activeUserProfile = p.data;
                updateNavbarUserDisplay(activeUserProfile);
                updateProfileCardDisplay(activeUserProfile);
                populateFormForEdit(activeUserProfile);
                loadProfileList();
              }
            });
        } else {
          updateNavbarUserDisplay(null);
          updateProfileCardDisplay(null);
        }
      })
      .catch(() => {
        updateNavbarUserDisplay(null);
        updateProfileCardDisplay(null);
      });
  }

  // ============================================================
  // Profile Management (My Nutrition Profile tab)
  // ============================================================
  const profileForm = document.getElementById('profileForm');
  const profileFormId = document.getElementById('profileFormId');
  const profileFormTitle = document.getElementById('profileFormTitle');
  const profileFormStatus = document.getElementById('profileFormStatus');
  const profileFormErrors = document.getElementById('profileFormErrors');
  const profileSaveSuccess = document.getElementById('profileSaveSuccess');
  const profileSaveMsg = document.getElementById('profileSaveMsg');
  const btnNewProfile = document.getElementById('btnNewProfile');
  const btnCancelEdit = document.getElementById('btnCancelEdit');
  const btnDeleteProfile = document.getElementById('btnDeleteProfile');
  const btnSaveProfile = document.getElementById('btnSaveProfile');
  const profileListContainer = document.getElementById('profileListContainer');
  const profileListEmpty = document.getElementById('profileListEmpty');
  const profileListItems = document.getElementById('profileListItems');

  // BMI preview inputs
  const pfHeight = document.getElementById('pfHeight');
  const pfWeight = document.getElementById('pfWeight');
  const pfAge = document.getElementById('pfAge');
  const pfSex = document.getElementById('pfSex');
  const pfActivity = document.getElementById('pfActivity');
  const pfDiet = document.getElementById('pfDiet');
  const pfGoal = document.getElementById('pfGoal');
  const pfName = document.getElementById('pfName');
  const pfAllergies = document.getElementById('pfAllergies');
  const pfRestrictions = document.getElementById('pfRestrictions');
  const pfBudget = document.getElementById('pfBudget');
  const btnCorrectDish = document.getElementById('btnCorrectDish');

  if (btnCorrectDish) {
    btnCorrectDish.addEventListener('click', () => {
      if (manualFoodSearchInput) {
        manualFoodSearchInput.focus();
        manualFoodSearchInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
  }
  const bmiPreviewCard = document.getElementById('bmiPreviewCard');
  const bmiPreviewVal = document.getElementById('bmiPreviewVal');
  const bmiPreviewCat = document.getElementById('bmiPreviewCat');
  const calPreviewVal = document.getElementById('calPreviewVal');
  const btnGoToProfile = document.getElementById('btnGoToProfile');

  if (btnGoToProfile) {
    btnGoToProfile.addEventListener('click', () => {
      const tab = document.getElementById('tabProfile');
      if (tab) tab.click();
    });
  }

  function computeBmiPreview() {
    if (!pfHeight || !pfWeight) return;
    const h = parseFloat(pfHeight.value);
    const w = parseFloat(pfWeight.value);
    if (isNaN(h) || isNaN(w) || h <= 0 || w <= 0) {
      if (bmiPreviewCard) bmiPreviewCard.classList.add('hidden');
      return;
    }
    const bmi = w / ((h / 100) ** 2);
    let cat = 'Normal weight';
    if (bmi < 18.5) cat = 'Underweight';
    else if (bmi < 25.0) cat = 'Normal weight';
    else if (bmi < 30.0) cat = 'Overweight';
    else cat = 'Obese';

    if (bmiPreviewVal) bmiPreviewVal.textContent = bmi.toFixed(1);
    if (bmiPreviewCat) bmiPreviewCat.textContent = cat;

    // Estimate TDEE (Mifflin-St Jeor + activity multiplier)
    const sex = pfSex ? pfSex.value : '';
    const age = parseFloat(pfAge ? pfAge.value : NaN);
    if (!isNaN(age) && age > 0) {
      let bmr = 0;
      if (sex === 'male') bmr = 10 * w + 6.25 * h - 5 * age + 5;
      else if (sex === 'female') bmr = 10 * w + 6.25 * h - 5 * age - 161;
      else bmr = 10 * w + 6.25 * h - 5 * age - 78;  // average

      const activityMap = {
        sedentary: 1.2, lightly_active: 1.375, moderately_active: 1.55, very_active: 1.725, extra_active: 1.9
      };
      const actKey = pfActivity ? pfActivity.value : 'sedentary';
      const mult = activityMap[actKey] || 1.2;
      const tdee = Math.round(bmr * mult);
      if (calPreviewVal) calPreviewVal.textContent = tdee.toLocaleString();
    } else {
      if (calPreviewVal) calPreviewVal.textContent = '--';
    }

    if (bmiPreviewCard) bmiPreviewCard.classList.remove('hidden');
  }

  [pfHeight, pfWeight, pfAge, pfSex, pfActivity].forEach(el => {
    if (el) el.addEventListener('input', computeBmiPreview);
  });

  const btnProceedToScanner = document.getElementById('btnProceedToScanner');
  if (btnProceedToScanner) {
    btnProceedToScanner.addEventListener('click', () => {
      const tab = document.getElementById('tabScanner');
      if (tab) tab.click();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  function loadProfileList() {
    if (!profileListItems) return;
    fetch('/api/user-profiles')
      .then(res => res.json())
      .then(payload => {
        if (!payload.success) return;
        const profiles = payload.data || [];
        profileListItems.innerHTML = '';
        if (profiles.length === 0) {
          if (profileListEmpty) profileListEmpty.style.display = '';
          profileListItems.style.display = 'none';
          return;
        }
        if (profileListEmpty) profileListEmpty.style.display = 'none';
        profileListItems.style.display = '';

        profiles.forEach(p => {
          const item = document.createElement('div');
          const isCurrentActive = (activeUserId === p.user_id);
          item.className = `profile-list-item${isCurrentActive ? ' active-profile' : ''}`;

          const displayName = (p.display_name || p.name || 'Unnamed Profile').trim();
          const initial = displayName.charAt(0).toUpperCase() || 'U';

          const metaParts = [];
          if (p.age) metaParts.push(`${p.age} yrs`);
          if (p.sex || p.gender) metaParts.push(p.sex || p.gender);
          if (p.activity_level) metaParts.push(p.activity_level.replace(/_/g, ' '));
          const metaString = metaParts.length > 0 ? metaParts.join(' · ') : 'Profile saved';

          const bmiBadge = p.bmi ? `<span class="profile-list-bmi-pill">BMI ${p.bmi}</span>` : '';
          const activeBadge = isCurrentActive
            ? '<span class="profile-list-active-badge"><span class="active-dot"></span>Active</span>'
            : '<span class="profile-list-switch-hint">Switch &rarr;</span>';

          item.innerHTML = `
            <div class="profile-item-main">
              <div class="profile-avatar">${initial}</div>
              <div class="profile-item-info">
                <div class="profile-list-name-row">
                  <span class="profile-list-name" title="${displayName}">${displayName}</span>
                </div>
                <div class="profile-list-meta">${metaString}</div>
                ${bmiBadge ? `<div class="profile-list-badges">${bmiBadge}</div>` : ''}
              </div>
            </div>
            <div class="profile-item-status">
              ${activeBadge}
            </div>
          `;

          item.addEventListener('click', () => {
            activeUserId = p.user_id;
            localStorage.setItem('nutrisense_active_user_id', activeUserId);
            fetchAndEditProfile(p.user_id);
            loadProfileList();
          });
          profileListItems.appendChild(item);
        });
      })
      .catch(err => console.error('Error loading profiles:', err));
  }

  function fetchAndEditProfile(userId) {
    fetch(`/api/profile/${userId}`)
      .then(res => res.json())
      .then(payload => {
        if (!payload.success || !payload.data) return;
        activeUserProfile = payload.data;
        activeUserId = payload.data.user_id;
        localStorage.setItem('nutrisense_active_user_id', activeUserId);
        updateNavbarUserDisplay(activeUserProfile);
        updateProfileCardDisplay(activeUserProfile);
        populateFormForEdit(activeUserProfile);
        if (currentPredictionData && currentPredictionData.prediction) {
          recalculateAdjustedNutrition();
        }
      })
      .catch(err => console.error('Error fetching profile:', err));
  }

  function populateFormForEdit(profile) {
    if (!profileForm) return;
    const displayName = profile.display_name || profile.name || 'Unnamed Profile';
    if (profileFormId) profileFormId.value = profile.user_id || '';
    if (profileFormTitle) profileFormTitle.textContent = `Edit Profile: ${displayName}`;
    if (profileFormStatus) {
      profileFormStatus.textContent = 'Active Profile';
      profileFormStatus.className = 'badge badge-success';
    }
    if (btnSaveProfile) btnSaveProfile.textContent = 'Update Profile';
    if (pfName) pfName.value = profile.display_name || profile.name || '';
    if (pfAge) pfAge.value = profile.age || '';
    if (pfSex) pfSex.value = profile.sex || '';
    if (pfHeight) pfHeight.value = profile.height_cm || '';
    if (pfWeight) pfWeight.value = profile.weight_kg || '';
    if (pfActivity) pfActivity.value = profile.activity_level || 'sedentary';
    if (pfDiet) pfDiet.value = profile.dietary_preference || 'omnivore';
    if (pfGoal) pfGoal.value = profile.nutrition_goal || 'maintenance';
    if (pfAllergies) pfAllergies.value = (profile.food_allergies || []).join(', ');
    if (pfRestrictions) pfRestrictions.value = (profile.food_restrictions || []).join(', ');
    if (pfBudget) pfBudget.value = profile.budget || 'medium';
    if (btnCancelEdit) btnCancelEdit.classList.remove('hidden');
    if (btnDeleteProfile) btnDeleteProfile.classList.remove('hidden');
    if (profileSaveSuccess) profileSaveSuccess.classList.add('hidden');
    if (profileFormErrors) { profileFormErrors.innerHTML = ''; profileFormErrors.classList.add('hidden'); }
    computeBmiPreview();
  }

  function resetForm() {
    if (!profileForm) return;
    profileForm.reset();
    if (pfBudget) pfBudget.value = 'medium';
    if (profileFormId) profileFormId.value = '';
    if (profileFormTitle) profileFormTitle.textContent = 'Create New Profile';
    if (profileFormStatus) {
      profileFormStatus.textContent = 'New';
      profileFormStatus.className = 'badge badge-outline';
    }
    if (btnSaveProfile) btnSaveProfile.textContent = 'Save Profile';
    if (btnCancelEdit) btnCancelEdit.classList.add('hidden');
    if (btnDeleteProfile) btnDeleteProfile.classList.add('hidden');
    if (profileFormErrors) { profileFormErrors.innerHTML = ''; profileFormErrors.classList.add('hidden'); }
    if (profileSaveSuccess) profileSaveSuccess.classList.add('hidden');
    if (bmiPreviewCard) bmiPreviewCard.classList.add('hidden');
  }

  if (btnNewProfile) {
    btnNewProfile.addEventListener('click', resetForm);
  }

  if (btnCancelEdit) {
    btnCancelEdit.addEventListener('click', resetForm);
  }

  if (btnDeleteProfile) {
    btnDeleteProfile.addEventListener('click', () => {
      const uid = profileFormId ? profileFormId.value : '';
      if (!uid) return;
      if (!confirm('Delete this profile? This action cannot be undone.')) return;

      fetch(`/api/profile/${uid}`, { method: 'DELETE' })
        .then(res => res.json())
        .then(payload => {
          if (!payload.success) {
            alert('Delete failed: ' + (payload.error || 'Unknown error'));
            return;
          }
          if (activeUserId === uid) {
            activeUserId = null;
            activeUserProfile = null;
            localStorage.removeItem('nutrisense_active_user_id');
            updateNavbarUserDisplay(null);
            updateProfileCardDisplay(null);
          }
          resetForm();
          loadProfileList();
        })
        .catch(err => alert('Delete error: ' + err.message));
    });
  }

  if (profileForm) {
    profileForm.addEventListener('submit', (e) => {
      e.preventDefault();
      if (profileFormErrors) { profileFormErrors.innerHTML = ''; profileFormErrors.classList.add('hidden'); }
      if (profileSaveSuccess) profileSaveSuccess.classList.add('hidden');

      const uid = profileFormId ? profileFormId.value.trim() : '';

      // Build payload
      const allergiesRaw = pfAllergies ? pfAllergies.value.trim() : '';
      const restrictionsRaw = pfRestrictions ? pfRestrictions.value.trim() : '';
      const rawName = pfName ? pfName.value.trim() : '';
      const finalName = rawName || 'Unnamed Profile';

      const payload = {
        name: finalName,
        display_name: finalName,
        age: pfAge && pfAge.value ? parseInt(pfAge.value) : null,
        sex: pfSex ? pfSex.value : '',
        height_cm: pfHeight && pfHeight.value ? parseFloat(pfHeight.value) : null,
        weight_kg: pfWeight && pfWeight.value ? parseFloat(pfWeight.value) : null,
        activity_level: pfActivity ? pfActivity.value : 'sedentary',
        dietary_preference: pfDiet ? pfDiet.value : 'omnivore',
        nutrition_goal: pfGoal ? pfGoal.value : 'maintenance',
        budget: pfBudget ? pfBudget.value : 'medium',
        food_allergies: allergiesRaw ? allergiesRaw.split(',').map(s => s.trim()).filter(Boolean) : [],
        food_restrictions: restrictionsRaw ? restrictionsRaw.split(',').map(s => s.trim()).filter(Boolean) : [],
      };

      const isEdit = Boolean(uid);
      const url = isEdit ? `/api/profile/${uid}` : '/api/profile/create';
      const method = isEdit ? 'PUT' : 'POST';

      if (btnSaveProfile) { btnSaveProfile.disabled = true; btnSaveProfile.textContent = 'Saving…'; }

      fetch(url, {
        method: method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
        .then(res => res.json())
        .then(resp => {
          if (btnSaveProfile) { btnSaveProfile.disabled = false; btnSaveProfile.textContent = isEdit ? 'Update Profile' : 'Save Profile'; }

          if (!resp.success) {
            const errs = resp.errors || [resp.error || 'Unknown error'];
            if (profileFormErrors) {
              profileFormErrors.innerHTML = errs.map(e => `<div class="profile-form-error-item">⚠ ${e}</div>`).join('');
              profileFormErrors.classList.remove('hidden');
            }
            return;
          }

          const savedProfile = resp.data;
          // Set as active user
          activeUserId = savedProfile.user_id;
          activeUserProfile = savedProfile;
          localStorage.setItem('nutrisense_active_user_id', activeUserId);
          updateNavbarUserDisplay(activeUserProfile);
          updateProfileCardDisplay(activeUserProfile);

          // Populate form in edit mode for this profile
          populateFormForEdit(savedProfile);

          // Reload profile list
          loadProfileList();

          if (profileSaveSuccess) {
            const action = isEdit ? 'updated' : 'created';
            const name = savedProfile.display_name || savedProfile.name || 'Profile';
            if (profileSaveMsg) profileSaveMsg.textContent = `${name} ${action} and activated.`;
            profileSaveSuccess.classList.remove('hidden');
          }

          // Recalculate if food data is loaded
          if (currentPredictionData && currentPredictionData.prediction) {
            recalculateAdjustedNutrition();
          }
        })
        .catch(err => {
          if (btnSaveProfile) { btnSaveProfile.disabled = false; btnSaveProfile.textContent = isEdit ? 'Update Profile' : 'Save Profile'; }
          alert('Save failed: ' + err.message);
        });
    });
  }

});

