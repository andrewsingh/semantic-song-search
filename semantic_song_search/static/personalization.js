class PersonalizationManager {
    constructor(app, api, analytics) {
        this.app = app;
        this.api = api;
        this.analytics = analytics;
    }

    // Personalization controls
    handleLambdaChange(value) {
        this.app.currentLambdaVal = value;
        const lambdaDisplay = this.app.domElements.lambdaValue;
        if (lambdaDisplay) {
            lambdaDisplay.textContent = value.toFixed(2);
            // Position above the lambda slider knob using accurate positioning
            const slider = this.app.domElements.lambdaSlider;
            this.positionSliderValue(slider, lambdaDisplay, value, 0, 1);
        }
        this.enableRerunButton();
        
        // Track lambda change
        this.analytics.trackEvent('Lambda Value Changed', {
            'lambda_val': value,
            'has_active_search': this.app.searchResults.length > 0
        });
    }

    handleFamiliarityMinChange(value) {
        const maxSlider = this.app.domElements.familiarityMax;
        const currentMax = parseFloat(maxSlider.value);
        
        // Ensure min doesn't exceed max
        if (value > currentMax) {
            value = currentMax;
            this.app.domElements.familiarityMin.value = value;
        }
        
        this.app.currentFamiliarityMin = value;
        this.updateFamiliarityRangeDisplay();
        this.enableRerunButton();
    }

    handleFamiliarityMaxChange(value) {
        const minSlider = this.app.domElements.familiarityMin;
        const currentMin = parseFloat(minSlider.value);
        
        // Ensure max doesn't go below min
        if (value < currentMin) {
            value = currentMin;
            this.app.domElements.familiarityMax.value = value;
        }
        
        this.app.currentFamiliarityMax = value;
        this.updateFamiliarityRangeDisplay();
        this.enableRerunButton();
    }

    updateFamiliarityRangeDisplay() {
        const minDisplay = document.getElementById('familiarity-min-value');
        const maxDisplay = document.getElementById('familiarity-max-value');
        
        if (minDisplay) {
            minDisplay.textContent = this.app.currentFamiliarityMin.toFixed(2);
            // Position above the min slider knob using accurate positioning
            const minSlider = this.app.domElements.familiarityMin;
            this.positionSliderValue(minSlider, minDisplay, this.app.currentFamiliarityMin, 0, 1);
        }
        if (maxDisplay) {
            maxDisplay.textContent = this.app.currentFamiliarityMax.toFixed(2);
            // Position above the max slider knob using accurate positioning
            const maxSlider = this.app.domElements.familiarityMax;
            this.positionSliderValue(maxSlider, maxDisplay, this.app.currentFamiliarityMax, 0, 1);
        }
    }

    // Accurately position slider value display above the slider thumb
    positionSliderValue(slider, valueDisplay, value, min, max) {
        if (!slider || !valueDisplay) return;
        
        // Calculate the percentage position of the value within the range
        const percent = (value - min) / (max - min);
        
        // Get the slider's bounding rectangle
        const sliderRect = slider.getBoundingClientRect();
        
        // Estimate thumb width (typically 20px for most browsers)
        const thumbWidth = 20;
        
        // Calculate the effective track width (slider width minus thumb width)
        const trackWidth = sliderRect.width - thumbWidth;
        
        // Calculate position: start at half thumb width, then add track progress
        const position = (thumbWidth / 2) + (percent * trackWidth);
        
        // Convert to percentage of total slider width
        const positionPercent = (position / sliderRect.width) * 100;
        
        valueDisplay.style.left = `${positionPercent}%`;
    }

    initializeSliderPositions() {
        // Initialize lambda slider position
        const lambdaDisplay = this.app.domElements.lambdaValue;
        if (lambdaDisplay) {
            const lambdaPercent = (this.app.currentLambdaVal - 0) / (1 - 0) * 100;
            lambdaDisplay.style.left = `${lambdaPercent}%`;
        }
        
        // Initialize familiarity range positions
        this.updateFamiliarityRangeDisplay();
    }

    enableRerunButton() {
        const rerunBtn = this.app.domElements.rerunSearchBtn;
        if (rerunBtn) {
            rerunBtn.disabled = false;
        }
    }

    disableRerunButton() {
        const rerunBtn = this.app.domElements.rerunSearchBtn;
        if (rerunBtn) {
            rerunBtn.disabled = true;
        }
    }

    async rerunSearchWithNewParameters() {
        
        // Save current values as the "active" values
        this.app.activeLambdaVal = this.app.currentLambdaVal;
        this.app.activeFamiliarityMin = this.app.currentFamiliarityMin;
        this.app.activeFamiliarityMax = this.app.currentFamiliarityMax;
        
        // Disable the rerun button
        this.disableRerunButton();
        
        // Track parameter update
        this.analytics.trackEvent('Personalization Parameters Updated', {
            'lambda_val': this.app.activeLambdaVal,
            'familiarity_min': this.app.activeFamiliarityMin,
            'familiarity_max': this.app.activeFamiliarityMax,
            'has_active_search': this.app.searchResults.length > 0
        });
        
        // Rerun the current search if we have one
        if (this.app.lastSearchRequestData) {
            await this.app.searchManager.handleSearch();
        }
    }

    // No-history weights
    initializeWeightInputs() {
        // Initialize input field values from JavaScript defaults
        const weightInputs = document.querySelectorAll('input[data-weight-name]');
        weightInputs.forEach(input => {
            const weightName = input.getAttribute('data-weight-name');
            if (weightName && this.app.currentNoHistoryWeights.hasOwnProperty(weightName)) {
                input.value = this.app.currentNoHistoryWeights[weightName];
            }
        });
    }

    handleNoHistoryWeightChange(inputId, value) {
        // Convert input ID to weight key
        const weightKey = inputId.replace('nh_', '');
        const convertedValue = parseFloat(value);

        // Validate the converted value
        if (isNaN(convertedValue)) {
            return; // Don't update if the value is invalid
        }

        // Store the current value
        this.app.currentNoHistoryWeights[weightKey] = convertedValue;

        // Check if any weights have changed from their active values
        this.updateNoHistoryRerunButtonState();
    }

    updateNoHistoryRerunButtonState() {
        const rerunBtn = this.app.domElements.noHistoryRerunBtn;
        if (!rerunBtn) return;

        // Check if any weight has changed from its active value
        let hasChanges = false;
        for (const [weightKey, currentValue] of Object.entries(this.app.currentNoHistoryWeights)) {
            const activeValue = this.app.activeNoHistoryWeights[weightKey];
            if (currentValue !== activeValue) {
                hasChanges = true;
                break;
            }
        }

        rerunBtn.disabled = !hasChanges;
    }

    async rerunSearchWithNoHistoryWeights() {
        // Save current values as the "active" values
        this.app.activeNoHistoryWeights = { ...this.app.currentNoHistoryWeights };

        // Disable the rerun button
        const rerunBtn = this.app.domElements.noHistoryRerunBtn;
        if (rerunBtn) {
            rerunBtn.disabled = true;
        }

        // Track weight update
        this.analytics.trackEvent('No-History Weights Updated', {
            'weights': this.app.activeNoHistoryWeights,
            'has_active_search': this.app.searchResults.length > 0
        });

        // Show confirmation that search was rerun
        this.showNoHistoryWeightsConfirmation();

        // Rerun the current search if we have one
        if (this.app.lastSearchRequestData) {
            await this.app.searchManager.handleSearch();
        }
    }

    resetNoHistoryWeightsToDefaults() {
        // Reset all no-history weight inputs to their default values
        // Top-level weights (a_i)
        if (this.app.domElements.nhA0SongSim && this.app.defaultNoHistoryWeights.a0_song_sim !== undefined) {
            this.app.domElements.nhA0SongSim.value = this.app.defaultNoHistoryWeights.a0_song_sim;
        }
        if (this.app.domElements.nhA1ArtistSim && this.app.defaultNoHistoryWeights.a1_artist_sim !== undefined) {
            this.app.domElements.nhA1ArtistSim.value = this.app.defaultNoHistoryWeights.a1_artist_sim;
        }
        if (this.app.domElements.nhA2TotalStreams && this.app.defaultNoHistoryWeights.a2_total_streams !== undefined) {
            this.app.domElements.nhA2TotalStreams.value = this.app.defaultNoHistoryWeights.a2_total_streams;
        }
        if (this.app.domElements.nhA3DailyStreams && this.app.defaultNoHistoryWeights.a3_daily_streams !== undefined) {
            this.app.domElements.nhA3DailyStreams.value = this.app.defaultNoHistoryWeights.a3_daily_streams;
        }
        if (this.app.domElements.nhA4ReleaseDate && this.app.defaultNoHistoryWeights.a4_release_date !== undefined) {
            this.app.domElements.nhA4ReleaseDate.value = this.app.defaultNoHistoryWeights.a4_release_date;
        }

        // Song descriptor weights (b_i)
        if (this.app.domElements.nhB0Genres && this.app.defaultNoHistoryWeights.b0_genres !== undefined) {
            this.app.domElements.nhB0Genres.value = this.app.defaultNoHistoryWeights.b0_genres;
        }
        if (this.app.domElements.nhB1VocalStyle && this.app.defaultNoHistoryWeights.b1_vocal_style !== undefined) {
            this.app.domElements.nhB1VocalStyle.value = this.app.defaultNoHistoryWeights.b1_vocal_style;
        }
        if (this.app.domElements.nhB2ProductionSoundDesign && this.app.defaultNoHistoryWeights.b2_production_sound_design !== undefined) {
            this.app.domElements.nhB2ProductionSoundDesign.value = this.app.defaultNoHistoryWeights.b2_production_sound_design;
        }
        if (this.app.domElements.nhB3LyricalMeaning && this.app.defaultNoHistoryWeights.b3_lyrical_meaning !== undefined) {
            this.app.domElements.nhB3LyricalMeaning.value = this.app.defaultNoHistoryWeights.b3_lyrical_meaning;
        }
        if (this.app.domElements.nhB4MoodAtmosphere && this.app.defaultNoHistoryWeights.b4_mood_atmosphere !== undefined) {
            this.app.domElements.nhB4MoodAtmosphere.value = this.app.defaultNoHistoryWeights.b4_mood_atmosphere;
        }
        if (this.app.domElements.nhB5Tags && this.app.defaultNoHistoryWeights.b5_tags !== undefined) {
            this.app.domElements.nhB5Tags.value = this.app.defaultNoHistoryWeights.b5_tags;
        }

        // Sync currentNoHistoryWeights with the reset values
        this.app.currentNoHistoryWeights = { ...this.app.defaultNoHistoryWeights };

        // Update the rerun button state since values may have changed
        this.updateNoHistoryRerunButtonState();

        // Track the reset action
        this.analytics.trackEvent('No-History Weights Reset', {
            'reset_to_defaults': this.app.defaultNoHistoryWeights,
            'has_active_search': this.app.searchResults.length > 0
        });
    }

    showNoHistoryWeightsConfirmation() {
        // Create a temporary confirmation message
        const rerunBtn = this.app.domElements.noHistoryRerunBtn;
        if (rerunBtn) {
            const originalText = rerunBtn.textContent;
            rerunBtn.textContent = 'Search Updated!';
            rerunBtn.style.background = 'rgba(29, 185, 84, 0.8)';

            setTimeout(() => {
                rerunBtn.textContent = originalText;
                rerunBtn.style.background = '';
            }, 2000);
        }
    }

    getActiveNoHistoryWeights() {
        // If we don't have active weights but have current weights, use current weights
        let weightsToUse = this.app.activeNoHistoryWeights;
        if (!weightsToUse || Object.keys(weightsToUse).length === 0) {
            weightsToUse = this.app.currentNoHistoryWeights;
        }

        return weightsToUse || {};
    }

    showPersonalizationControls(hasHistory) {
        const controls = document.getElementById('personalization-controls');
        const resultsRight = document.querySelector('.results-right');
        const topArtistsFilterOption = document.getElementById('top-artists-filter-option');
        const noHistoryWeightsSection = this.app.domElements.noHistoryWeightsSection;

        if (controls && resultsRight) {
            this.app.hasPersonalizationHistory = hasHistory;

            if (hasHistory) {
                // Move personalization controls to results-right area
                controls.style.display = 'flex';
                resultsRight.appendChild(controls);

                // Initialize slider value positions
                this.initializeSliderPositions();

                // Hide no-history weights section in history mode
                if (noHistoryWeightsSection) {
                    noHistoryWeightsSection.style.display = 'none';
                }

                // Hide top artists filter in history mode
                if (topArtistsFilterOption) {
                    topArtistsFilterOption.style.display = 'none';
                }
            } else {
                // Hide personalization controls in no-history mode
                controls.style.display = 'none';

                // Show no-history weights section in no-history mode
                if (noHistoryWeightsSection) {
                    noHistoryWeightsSection.style.display = 'block';
                }

                // Show top artists filter in no-history mode
                if (topArtistsFilterOption) {
                    topArtistsFilterOption.style.display = 'block';
                }
            }
        }
    }

    displayRankingWeights(rankingWeights) {
        let rankingWeightsContainer = document.getElementById('ranking-weights-container');
        if (rankingWeightsContainer) {
            rankingWeightsContainer.style.display = 'none';
        }
    }
}
