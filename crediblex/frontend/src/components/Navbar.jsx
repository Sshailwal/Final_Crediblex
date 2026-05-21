// components/Navbar.jsx

export default function Navbar() {
  return (
    <nav
      style={{
        height: "64px",
        background: "#ffffff",
        borderBottom: "1px solid #e5e7eb",

        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",

        padding: "0 24px",

        position: "sticky",
        top: 0,
        zIndex: 100,
      }}
    >
      {/* Left Logo */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          minWidth: "220px",
        }}
      >
        <div
          style={{
            width: "30px",
            height: "30px",
            borderRadius: "8px",
            background: "linear-gradient(135deg, #ef4444, #dc2626)",

            display: "flex",
            alignItems: "center",
            justifyContent: "center",

            color: "white",
            fontWeight: "700",
            fontSize: "14px",
          }}
        >
          C
        </div>

        <span
          style={{
            fontSize: "16px",
            fontWeight: "700",
            color: "#111827",
          }}
        >
          CredibleX
        </span>
      </div>

      {/* Center Title */}
      <div
        style={{
          flex: 1,
          textAlign: "center",
          fontSize: "15px",
          fontWeight: "600",
          color: "#1f2937",
        }}
      >
        News Article Credibility Dashboard
      </div>

      {/* Right Side */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          gap: "18px",
          minWidth: "220px",
        }}
      >
        

        {/* User */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <div
            style={{
              width: "28px",
              height: "28px",
              borderRadius: "999px",
              background: "#fee2e2",

              display: "flex",
              alignItems: "center",
              justifyContent: "center",

              color: "#ef4444",
              fontSize: "13px",
              fontWeight: "700",
            }}
          >
            👤
          </div>

          <span
            style={{
              fontSize: "14px",
              fontWeight: "500",
              color: "#374151",
            }}
          >
            User
          </span>
        </div>
      </div>
    </nav>
  );
}