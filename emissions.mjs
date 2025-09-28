const DEFAULT_FACTORS = {
  organic_food_scraps: { emission_factor: '451baec4-3aa4-4462-8b4a-6436ac147346' },
  mixed_plastic: { emission_factor: 'c9d3c74f-ff9b-4c19-858c-786d1ac6df43' },
  single_use_plastic: { emission_factor: 'c9d3c74f-ff9b-4c19-858c-786d1ac6df43' },
  paper_cardboard: { emission_factor: 'bf40a5ee-6833-4d4c-b7e9-e0d581aba3a4' },
  aluminum_can: { emission_factor: '33df28d6-1b08-44a8-9b86-6d8af1e92786' },
  glass_bottle: { emission_factor: '9b8b42a2-8a9c-496c-a920-05d4953ea055' },
  textile: { emission_factor: 'f9627b5f-77b4-4605-9e1c-207d1f5e792d' },
  electronics: { emission_factor: 'd0122ce0-44df-43b4-bac7-ccb316f52c57' },
  battery: { emission_factor: '1e4f00df-9074-4d77-b492-1be917986996' },
  compostable_packaging: { emission_factor: 'bf40a5ee-6833-4d4c-b7e9-e0d581aba3a4' },
  other: { emission_factor: 'c27e40d5-12f8-4aad-95e7-45449391cc55' },
};

const CLIMATIQ_ENDPOINT = 'https://beta3.api.climatiq.io/estimate';

export const handler = async (event) => {
  const category = event.queryStringParameters?.category || 'other';
  const factor = DEFAULT_FACTORS[category] || DEFAULT_FACTORS.other;

  if (!process.env.CLIMATIQ_API_KEY) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Missing Climatiq API key' }),
    };
  }

  try {
    const response = await fetch(CLIMATIQ_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.CLIMATIQ_API_KEY}`,
      },
      body: JSON.stringify({
        emission_factor: factor,
        parameters: {
          energy: 1,
          energy_unit: 'kWh',
        },
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Climatiq error: ${text}`);
    }

    const data = await response.json();
    const kgCO2e = data.co2e;
    const pointsAwarded = Math.max(0, Math.round((3 - kgCO2e) * 10));

    return {
      statusCode: 200,
      body: JSON.stringify({
        category,
        kgCO2e,
        pointsAwarded,
      }),
    };
  } catch (error) {
    console.error('Emission lookup failed', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Failed to estimate emissions', details: error.message }),
    };
  }
};