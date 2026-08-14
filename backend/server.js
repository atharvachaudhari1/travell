import express from "express";
const app = express();
app.use(express.json());
app.get("/", (_req, res) => res.json({ ok: true }));
const PORT = process.env.PORT || 3000;
app.listen(PORT);
// cors middleware added