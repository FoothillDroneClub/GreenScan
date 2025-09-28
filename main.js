import { getEmissionEstimate, loadMobileNet, classifyImage } from './utils.js';
import { initializeLeaderboard, recordUserScore } from './mockLeaderboard.js';

const videoFeed = document.getElementById('videoFeed');
const captureCanvas = document.getElementById('captureCanvas');
const captureButton = document.getElementById('captureButton');
const uploadInput = document.getElementById('uploadInput');
const statusMessage = document.getElementById('statusMessage');
const analysisCard = document.getElementById('analysisCard');
const itemNameDisplay = document.getElementById('itemName');
const itemCategoryDisplay = document.getElementById('itemCategory');
const itemEmissionDisplay = document.getElementById('itemEmission');
const pointsEarnedDisplay = document.getElementById('pointsEarned');
const totalPointsDisplay = document.getElementById('totalPoints');
const badgeDisplay = document.getElementById('badgeDisplay');
const leaderboardList = document.getElementById('leaderboardList');

const localState = {
  totalPoints: Number(localStorage.getItem('totalPoints') || 0),
  classifier: null,
};

document.addEventListener('DOMContentLoaded', async () => {
  totalPointsDisplay.textContent = localState.totalPoints;
  updateBadge(localState.totalPoints);
  await initializeCamera();
  await warmupClassifier();
  renderLeaderboard();
});

captureButton.addEventListener('click', async () => {
  statusMessage.textContent = 'Capturing photo…';
  const imageBlob = await grabFrameFromVideo();
  if (!imageBlob) {
    statusMessage.textContent = 'Unable to capture image. Try uploading instead.';
    return;
  }
  await handleImageAnalysis(imageBlob);
});

uploadInput.addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  statusMessage.textContent = 'Uploading photo…';
  await handleImageAnalysis(file);
});

async function warmupClassifier() {
  try {
    statusMessage.textContent = 'Loading trash classifier…';
    localState.classifier = await loadMobileNet();
    statusMessage.textContent = 'Aim at a piece of trash to classify it.';
  } catch (error) {
    console.error('Classifier load failed', error);
    statusMessage.textContent = 'Could not load classifier. Try refreshing the page.';
  }
}

async function initializeCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    videoFeed.srcObject = stream;
    statusMessage.textContent = 'Aim at a piece of trash to classify it.';
  } catch (error) {
    console.error('Camera init failed', error);
    statusMessage.textContent = 'Camera access blocked. Upload a photo instead.';
    uploadInput.style.display = 'block';
  }
}

async function grabFrameFromVideo() {
  if (!videoFeed.srcObject) return null;
  const track = videoFeed.srcObject.getVideoTracks()[0];
  const capture = new ImageCapture(track);
  try {
    return await capture.takePhoto();
  } catch (error) {
    console.warn('takePhoto failed, fallback to canvas', error);
    const video = videoFeed;
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const context = captureCanvas.getContext('2d');
    context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    return await new Promise((resolve) => captureCanvas.toBlob(resolve, 'image/jpeg', 0.9));
  }
}

async function handleImageAnalysis(imageBlob) {
  try {
    if (!localState.classifier) {
      await warmupClassifier();
    }
    statusMessage.textContent = 'Analyzing trash photo…';

    const analysis = await classifyImage(localState.classifier, imageBlob);
    if (!analysis) throw new Error('No match found');

    const { displayName, category } = analysis;
    const emissionData = await getEmissionEstimate(category);

    updateAnalysisUI(displayName, emissionData);
    persistPoints(emissionData.pointsAwarded);
    await recordUserScore('You', localState.totalPoints);
    renderLeaderboard();
  } catch (error) {
    console.error('Analysis failed', error);
    statusMessage.textContent = 'Could not classify item. Try better lighting or a different angle.';
  }
}

function updateAnalysisUI(name, emissionData) {
  const { category, kgCO2e, pointsAwarded } = emissionData;
  analysisCard.classList.remove('hidden');
  itemNameDisplay.textContent = name;
  itemCategoryDisplay.textContent = `Category: ${category}`;
  itemEmissionDisplay.textContent = `Estimated CO₂e: ${kgCO2e.toFixed(2)} kg`;
  pointsEarnedDisplay.textContent = `Eco Points Earned: ${pointsAwarded}`;
  statusMessage.textContent = pointsAwarded > 0
    ? 'Nice! You kept emissions low and earned points.'
    : 'High emissions detected. Consider greener alternatives next time.';
}

function persistPoints(points) {
  localState.totalPoints = Math.max(localState.totalPoints + points, 0);
  localStorage.setItem('totalPoints', localState.totalPoints);
  totalPointsDisplay.textContent = localState.totalPoints;
  updateBadge(localState.totalPoints);
}

function updateBadge(totalPoints) {
  let badge = '🌱 Starter';
  if (totalPoints >= 200) badge = '🌎 Guardian';
  else if (totalPoints >= 100) badge = '🌿 Trailblazer';
  else if (totalPoints >= 50) badge = '🍃 Spark';
  badgeDisplay.textContent = badge;
}

function renderLeaderboard() {
  const entries = initializeLeaderboard();
  leaderboardList.innerHTML = '';
  entries.forEach((entry) => {
    const item = document.createElement('li');
    item.className = 'leaderboard-item';
    item.innerHTML = `
      <span class="name">${entry.name}</span>
      <span class="points">${entry.points} pts</span>
    `;
    leaderboardList.appendChild(item);
  });
}