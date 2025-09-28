'use strict';

const CATEGORY_KEYWORDS = [
  { keywords: ['banana peel', 'apple core', 'food scrap', 'leftovers', 'compost'], category: 'organic_food_scraps', estimated_kg_co2: 0.2 },
  { keywords: ['paper cup', 'cardboard', 'paper bag', 'newspaper', 'magazine'], category: 'paper_cardboard', estimated_kg_co2: 0.5 },
  { keywords: ['plastic bottle', 'water bottle', 'pet bottle', 'plastic cup'], category: 'single_use_plastic', estimated_kg_co2: 3.3 },
  { keywords: ['plastic wrapper', 'plastic packaging', 'plastic container'], category: 'mixed_plastic', estimated_kg_co2: 2.8 },
  { keywords: ['aluminum can', 'soda can', 'beer can'], category: 'aluminum_can', estimated_kg_co2: 1.6 },
  { keywords: ['glass bottle', 'wine bottle', 'glass jar'], category: 'glass_bottle', estimated_kg_co2: 1.2 },
  { keywords: ['t shirt', 'jeans', 'cloth', 'fabric', 'sock'], category: 'textile', estimated_kg_co2: 5.0 },
  { keywords: ['phone', 'laptop', 'charger', 'tablet'], category: 'electronics', estimated_kg_co2: 10.0 },
  { keywords: ['battery', 'aa battery', 'aaa battery'], category: 'battery', estimated_kg_co2: 3.8 },
  { keywords: ['compostable cup', 'bioplastic', 'compostable packaging'], category: 'compostable_packaging', estimated_kg_co2: 0.6 },
];

const MOBILENET_TOPK = 5;

function matchKeyword(description) {
  const normalized = description.toLowerCase();
  for (const entry of CATEGORY_KEYWORDS) {
    if (entry.keywords.some((keyword) => normalized.includes(keyword))) {
      return entry;
    }
  }
  return { category: 'other', estimated_kg_co2: 2.5 };
}

export async function loadMobileNet() {
  if (!window.mobilenet) {
    throw new Error('MobileNet library missing');
  }
  const model = await window.mobilenet.load();
  return model;
}

export async function classifyImage(model, imageBlob) {
  const imageElement = await createImageElement(imageBlob);
  const predictions = await model.classify(imageElement, MOBILENET_TOPK);
  if (!predictions || predictions.length === 0) {
    return null;
  }

  const bestMatch = predictions.find((pred) => pred.probability >= 0.2) || predictions[0];
  const keywordMatch = matchKeyword(bestMatch.className || bestMatch.label || '');

  return {
    displayName: bestMatch.className || bestMatch.label || 'Unknown item',
    category: keywordMatch.category,
    estimatedKg: keywordMatch.estimated_kg_co2,
  };
}

async function createImageElement(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = reader.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export async function getEmissionEstimate(category) {
  const response = await fetch(`/.netlify/functions/emissions?category=${encodeURIComponent(category)}`);
  if (!response.ok) throw new Error('Failed to fetch emissions');
  const data = await response.json();
  return data;
}