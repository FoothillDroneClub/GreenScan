'use strict';

const STORAGE_KEY = 'eco_leaderboard';

const seedData = [
  { name: 'You', points: Number(localStorage.getItem('totalPoints') || 0) },
  { name: 'Ava', points: 180 },
  { name: 'Noah', points: 120 },
  { name: 'Mia', points: 85 },
  { name: 'Leo', points: 60 },
];

export function initializeLeaderboard() {
  const existing = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
  if (!existing) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(seedData));
    return seedData;
  }
  return existing.sort((a, b) => b.points - a.points).slice(0, 5);
}

export async function recordUserScore(name, points) {
  const entries = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  const currentUser = entries.find((entry) => entry.name === name);
  if (currentUser) {
    currentUser.points = points;
  } else {
    entries.push({ name, points });
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}