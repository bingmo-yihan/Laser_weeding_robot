import mysql from 'mysql2/promise';

export default async function handler(req, res) {
  // 处理跨域预检请求
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const { name, phone, company, content } = req.body;

  try {
    const connection = await mysql.createConnection({
      host: "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
      user: "g7E7kFFXdw5cWZy.root",
      password: "q5skQ4NCT1M3PJYA",
      database: "zhinong",
      port: 4000,
      ssl: { rejectUnauthorized: false }
    });

    const [result] = await connection.execute(
      "INSERT INTO contact (name, phone, company, content, create_time) VALUES (?, ?, ?, ?, NOW())",
      [name, phone, company || "无", content]
    );

    await connection.end();

    res.setHeader('Access-Control-Allow-Origin', '*');
    return res.status(200).json({ success: true, insertId: result.insertId });

  } catch (err) {
    console.error('TiDB Error:', err);
    return res.status(500).json({ success: false, error: err.message });
  }
}