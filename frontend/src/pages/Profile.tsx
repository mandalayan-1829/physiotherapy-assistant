import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { updateProfile } from "@/services/userService";
import Card from "@/components/common/Card";

export default function Profile() {
  const { user, setUser } = useAuth();
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({
    name: user?.name || "",
    age: user?.age || 0,
    gender: user?.gender || "",
    height_cm: user?.height_cm || 0,
    weight_kg: user?.weight_kg || 0,
    blood_group: user?.blood_group || "",
    medical_conditions: user?.medical_conditions || "",
    exercise_limitations: user?.exercise_limitations || "",
    rehab_goals: user?.rehab_goals || "",
  });

  const handleSave = async () => {
    try {
      const updated = await updateProfile(form);
      setUser(updated);
      setEditing(false);
    } catch (err) {
      console.error("Failed to update profile:", err);
    }
  };

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <h1 className="text-3xl font-extrabold text-surface-900 mb-2">My Profile</h1>
      <p className="text-surface-500 mb-8">Manage your account and therapy settings</p>

      {/* Profile Summary */}
      <Card className="mb-8">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 bg-primary-500 rounded-2xl flex items-center justify-center text-2xl font-bold text-white">
            {user?.name?.charAt(0) || "U"}
          </div>
          <div>
            <h2 className="text-xl font-bold text-surface-900">{user?.name}</h2>
            <p className="text-sm text-surface-500">{user?.email}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-surface-50 rounded-xl">
            <p className="text-xs text-surface-400 uppercase tracking-wider">Age</p>
            <p className="font-bold text-surface-900">{user?.age || "—"}</p>
          </div>
          <div className="text-center p-3 bg-surface-50 rounded-xl">
            <p className="text-xs text-surface-400 uppercase tracking-wider">Height</p>
            <p className="font-bold text-surface-900">{user?.height_cm ? `${user.height_cm} cm` : "—"}</p>
          </div>
          <div className="text-center p-3 bg-surface-50 rounded-xl">
            <p className="text-xs text-surface-400 uppercase tracking-wider">Weight</p>
            <p className="font-bold text-surface-900">{user?.weight_kg ? `${user.weight_kg} kg` : "—"}</p>
          </div>
          <div className="text-center p-3 bg-surface-50 rounded-xl">
            <p className="text-xs text-surface-400 uppercase tracking-wider">Blood</p>
            <p className="font-bold text-surface-900">{user?.blood_group || "—"}</p>
          </div>
        </div>
      </Card>

      {/* Edit Form */}
      <Card>
        <div className="flex items-center justify-between mb-6">
          <h3 className="font-bold text-surface-900">Personal Information</h3>
          <button
            onClick={() => editing ? handleSave() : setEditing(true)}
            className={`px-4 py-2 text-sm font-medium rounded-xl transition-all ${
              editing
                ? "bg-primary-500 text-white hover:bg-primary-600"
                : "bg-surface-100 text-surface-700 hover:bg-surface-200"
            }`}
          >
            {editing ? "Save Changes" : "Edit Profile"}
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-5">
          {[
            { label: "Full Name", key: "name", type: "text" },
            { label: "Age", key: "age", type: "number" },
            { label: "Gender", key: "gender", type: "text" },
            { label: "Blood Group", key: "blood_group", type: "text" },
            { label: "Height (cm)", key: "height_cm", type: "number" },
            { label: "Weight (kg)", key: "weight_kg", type: "number" },
          ].map((field) => (
            <div key={field.key}>
              <label className="block text-sm font-medium text-surface-600 mb-1.5">
                {field.label}
              </label>
              <input
                type={field.type}
                value={(form as any)[field.key] || ""}
                onChange={(e) => setForm({ ...form, [field.key]: field.type === "number" ? Number(e.target.value) : e.target.value })}
                disabled={!editing}
                className="w-full px-4 py-2.5 rounded-xl border border-surface-200 text-sm disabled:bg-surface-50 disabled:text-surface-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          ))}
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-surface-600 mb-1.5">
              Medical Conditions
            </label>
            <textarea
              value={form.medical_conditions}
              onChange={(e) => setForm({ ...form, medical_conditions: e.target.value })}
              disabled={!editing}
              rows={3}
              className="w-full px-4 py-2.5 rounded-xl border border-surface-200 text-sm disabled:bg-surface-50 disabled:text-surface-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-surface-600 mb-1.5">
              Exercise Limitations
            </label>
            <textarea
              value={form.exercise_limitations}
              onChange={(e) => setForm({ ...form, exercise_limitations: e.target.value })}
              disabled={!editing}
              rows={2}
              className="w-full px-4 py-2.5 rounded-xl border border-surface-200 text-sm disabled:bg-surface-50 disabled:text-surface-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-surface-600 mb-1.5">
              Rehab Goals
            </label>
            <textarea
              value={form.rehab_goals}
              onChange={(e) => setForm({ ...form, rehab_goals: e.target.value })}
              disabled={!editing}
              rows={2}
              className="w-full px-4 py-2.5 rounded-xl border border-surface-200 text-sm disabled:bg-surface-50 disabled:text-surface-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>
      </Card>
    </div>
  );
}
