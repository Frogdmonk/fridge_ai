
        var jobId = "{{ job_id }}";
        var statusMessages = {
            detect: "Scanning keyframes for ingredients...",
            embed: "Converting ingredients into a search vector...",
            retrieve: "Searching 400+ recipes for the closest matches...",
            generate: "Asking the LLM to write your recipe..."
        };

        var excludedIngredients = new Set();

        function showToast(message, type) {
            var container = document.getElementById("toastContainer");
            var toast = document.createElement("div");
            toast.className = "toast " + (type || "info");
            toast.textContent = message;
            container.appendChild(toast);
            setTimeout(function() {
                toast.classList.add("toast-out");
                setTimeout(function() { toast.remove(); }, 300);
            }, 3000);
        }

        function renderSteps(steps) {
            var stepper = document.getElementById("stepper");
            stepper.innerHTML = "";

            for (var i = 0; i < steps.length; i++) {
                var step = steps[i];
                var node = document.createElement("div");
                node.className = "step-node " + step.status;

                var circle = document.createElement("div");
                circle.className = "step-circle";
                circle.textContent = step.status === "done" ? "OK" : (i + 1);
                if (step.status === "done") { circle.innerHTML = "&#10003;"; }

                var label = document.createElement("div");
                label.className = "step-label";
                label.textContent = step.label;

                node.appendChild(circle);
                node.appendChild(label);
                stepper.appendChild(node);

                if (i < steps.length - 1) {
                    var line = document.createElement("div");
                    line.className = "step-line " + (step.status === "done" ? "done" : "");
                    stepper.appendChild(line);
                }

                if (step.status === "active" && statusMessages[step.key]) {
                    document.getElementById("statusLine").textContent = statusMessages[step.key];
                }
            }
        }

        function renderLogs(logs) {
            var logsBox = document.getElementById("logs");
            var html = "";
            for (var i = 0; i < logs.length; i++) {
                html += "<div class=\"log-line\">" + logs[i] + "</div>";
            }
            logsBox.innerHTML = html;
            logsBox.scrollTop = logsBox.scrollHeight;
        }

        function confidenceColor(conf) {
            if (conf >= 0.75) { return "#4ade80"; }
            if (conf >= 0.55) { return "#facc15"; }
            return "#fb923c";
        }

        function setupTabs() {
            var buttons = document.querySelectorAll(".tab-btn");
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].addEventListener("click", function() {
                    var allButtons = document.querySelectorAll(".tab-btn");
                    var allPanels = document.querySelectorAll(".tab-panel");
                    for (var j = 0; j < allButtons.length; j++) { allButtons[j].classList.remove("active"); }
                    for (var j = 0; j < allPanels.length; j++) { allPanels[j].classList.remove("active"); }
                    this.classList.add("active");
                    document.getElementById("tab-" + this.dataset.tab).classList.add("active");
                });
            }
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        function formatRecipe(text) {
            var escaped = escapeHtml(text);
            escaped = escaped.replace(/^### (.*)$/gm, "<h4>$1</h4>");
            escaped = escaped.replace(/^## (.*)$/gm, "<h3>$1</h3>");
            escaped = escaped.replace(/^# (.*)$/gm, "<h2>$1</h2>");
            escaped = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
            escaped = escaped.replace(/^\d+\.\s+(.*)$/gm, "<li>$1</li>");
            escaped = escaped.replace(/(<li>.*<\/li>)/gs, "<ol>$1</ol>");
            escaped = escaped.replace(/\n{2,}/g, "</p><p>");
            escaped = "<p>" + escaped + "</p>";
            return escaped;
        }

        function renderResult(result) {
            document.getElementById("statusLine").textContent = "Done.";
            var resultBox = document.getElementById("result");
            resultBox.classList.remove("hidden");

            if (result.overview_image && result.overview_count > 1) {
                var panel = document.getElementById("overviewPanel");
                panel.classList.remove("hidden");
                document.getElementById("overviewImage").src = result.overview_image;
                document.getElementById("overviewCaption").textContent =
                    result.overview_count + " ingredients detected together in a single frame";
            }

            var grid = document.getElementById("detectionGrid");
            grid.innerHTML = "";

            if (result.detections.length === 0) {
                grid.innerHTML = "<p class=\"muted\">No food items detected.</p>";
            }

            for (var i = 0; i < result.detections.length; i++) {
                var item = result.detections[i];
                var pct = Math.round(item.confidence * 100);
                var isExcluded = excludedIngredients.has(item.name);

                var card = document.createElement("div");
                card.className = "detection-card reveal" + (isExcluded ? " excluded" : "");
                card.style.animationDelay = (i * 90) + "ms";
                card.style.setProperty("--tag-color", item.color || "#4ade80");
                card.innerHTML =
                    "<div class=\"detection-img-wrap\"><img src=\"" + item.image + "\"><span class=\"color-tag\"></span></div>" +
                    "<div class=\"detection-info\">" +
                        "<label class=\"toggle-row\">" +
                            "<input type=\"checkbox\" class=\"ingredient-toggle\" data-name=\"" + item.name + "\" " + (isExcluded ? "" : "checked") + ">" +
                            "<span class=\"detection-name\">" + item.name + "</span>" +
                        "</label>" +
                        "<div class=\"confidence-bar\"><div class=\"confidence-fill\" style=\"width:" + pct + "%; background:" + confidenceColor(item.confidence) + "\"></div></div>" +
                        "<span class=\"detection-meta\">" + pct + "% confidence, " + item.hit_count + " frames</span>" +
                    "</div>";
                grid.appendChild(card);
            }

            var toggles = document.querySelectorAll(".ingredient-toggle");
            for (var i = 0; i < toggles.length; i++) {
                toggles[i].addEventListener("change", function() {
                    var name = this.dataset.name;
                    var card = this.closest(".detection-card");
                    if (this.checked) {
                        excludedIngredients.delete(name);
                        card.classList.remove("excluded");
                    } else {
                        excludedIngredients.add(name);
                        card.classList.add("excluded");
                    }
                });
            }

            if (result.detections.length > 1) {
                document.getElementById("regenerateBtn").classList.remove("hidden");
            }

            var recipeList = document.getElementById("recipeList");
            recipeList.innerHTML = "";
            for (var i = 0; i < result.recipes.length; i++) {
                var r = result.recipes[i];
                var pill = document.createElement("div");
                pill.className = "recipe-pill reveal";
                pill.style.animationDelay = (i * 100) + "ms";
                pill.innerHTML =
                    "<div class=\"pill-header\"><span>" + r.title + "</span><span class=\"pill-score\">" + Math.round(r.score * 100) + "%</span></div>" +
                    "<div class=\"pill-body hidden\"><pre>" + escapeHtml(r.text) + "</pre></div>";
                (function(pillEl) {
                    pillEl.querySelector(".pill-header").addEventListener("click", function() {
                        pillEl.querySelector(".pill-body").classList.toggle("hidden");
                    });
                })(pill);
                recipeList.appendChild(pill);
            }

            var missingPanel = document.getElementById("missingPanel");
            var missingList = document.getElementById("missingList");
            missingList.innerHTML = "";
            if (result.missing_ingredients && result.missing_ingredients.length > 0) {
                missingPanel.style.display = "block";
                for (var i = 0; i < result.missing_ingredients.length; i++) {
                    var chip = document.createElement("span");
                    chip.className = "missing-chip";
                    chip.textContent = result.missing_ingredients[i];
                    missingList.appendChild(chip);
                }
            } else {
                missingPanel.style.display = "none";
            }

            document.getElementById("finalRecipe").innerHTML = formatRecipe(result.final_recipe);
            document.getElementById("promptView").textContent = result.prompt || "";
        }

        document.getElementById("regenerateBtn").addEventListener("click", function() {
            var btn = this;
            btn.disabled = true;
            btn.textContent = "Regenerating...";

            var excludedArray = Array.from(excludedIngredients);

            fetch("/regenerate/" + jobId, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ excluded: excludedArray })
            })
            .then(function(res) {
                return res.json().then(function(data) { return { ok: res.ok, data: data }; });
            })
            .then(function(result) {
                if (!result.ok) {
                    showToast(result.data.error || "Regeneration failed", "error");
                } else {
                    renderResult(result.data);
                    showToast("Recipe updated based on your corrections", "success");
                    document.querySelector(".tab-btn[data-tab='recipe']").click();
                }
            })
            .catch(function() {
                showToast("Something went wrong. Try again.", "error");
            })
            .finally(function() {
                btn.disabled = false;
                btn.textContent = "Regenerate Recipe";
            });
        });

        function poll() {
            fetch("/progress/" + jobId)
                .then(function(res) { return res.json(); })
                .then(function(data) {
                    if (data.error) {
                        var box = document.getElementById("errorBox");
                        box.textContent = "Error: " + data.error;
                        box.classList.remove("hidden");
                    }

                    renderSteps(data.steps);
                    renderLogs(data.logs);

                    if (data.finished) {
                        if (data.result) {
                            renderResult(data.result);
                        }
                        return;
                    }

                    setTimeout(poll, 800);
                })
                .catch(function(err) {
                    console.error("Polling error:", err);
                    setTimeout(poll, 1500);
                });
        }

        setupTabs();
        poll();

        var canvas = document.getElementById("particles");
        var ctx = canvas.getContext("2d");
        var particles = [];

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();

        for (var i = 0; i < 40; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                r: Math.random() * 1.8 + 0.4,
                vx: (Math.random() - 0.5) * 0.15,
                vy: (Math.random() - 0.5) * 0.15,
                alpha: Math.random() * 0.4 + 0.1
            });
        }

        function animateParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (var i = 0; i < particles.length; i++) {
                var p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0) { p.x = canvas.width; }
                if (p.x > canvas.width) { p.x = 0; }
                if (p.y < 0) { p.y = canvas.height; }
                if (p.y > canvas.height) { p.y = 0; }
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(120,150,255," + p.alpha + ")";
                ctx.fill();
            }
            requestAnimationFrame(animateParticles);
        }
        animateParticles();
    