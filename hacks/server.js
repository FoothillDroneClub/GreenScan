import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import emissionsRouter from './api/emissions.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5173;

app.use(express.static(path.join(__dirname)));
app.use('/api', emissionsRouter);

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});