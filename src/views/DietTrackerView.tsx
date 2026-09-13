import { useState } from 'react';
import { 
  ChevronDown, 
  ChevronRight, 
  Droplet, 
  Egg, 
  Flame, 
  Plus, 
  Trash2, 
  Utensils, 
  Wheat 
} from 'lucide-react';
import { DietEntry } from '../types';

interface DietTrackerViewProps {
  dietEntries: DietEntry[];
  onAddEntry: (entry: Omit<DietEntry, 'id'>) => void;
  onDeleteEntry: (id: number) => void;
}

export function DietTrackerView({ dietEntries, onAddEntry, onDeleteEntry }: DietTrackerViewProps) {
  const [mealName, setMealName] = useState<string>('');
  const [calories, setCalories] = useState<string>('400');
  const [protein, setProtein] = useState<string>('30');
  const [carbs, setCarbs] = useState<string>('40');
  const [fats, setFats] = useState<string>('12');
  const [showAddForm, setShowAddForm] = useState<boolean>(false);

  // Expandable sections state
  const [openSections, setOpenSections] = useState<{
    macros: boolean;
    meals: boolean;
  }>({
    macros: true,
    meals: true,
  });

  const toggleSection = (key: keyof typeof openSections) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Quick preset templates
  const presets = [
    { name: 'Whey Protein Shake + Banana', cal: 320, p: 35, c: 38, f: 3 },
    { name: 'Grilled Chicken & Quinoa Salad', cal: 520, p: 45, c: 42, f: 14 },
    { name: 'Eggs, Avocado Toast & Spinach', cal: 460, p: 24, c: 32, f: 22 },
    { name: 'Greek Yogurt with Blueberries & Honey', cal: 260, p: 22, c: 30, f: 4 },
  ];

  const applyPreset = (p: typeof presets[0]) => {
    setMealName(p.name);
    setCalories(p.cal.toString());
    setProtein(p.p.toString());
    setCarbs(p.c.toString());
    setFats(p.f.toString());
    setShowAddForm(true);
  };

  const handleAddMeal = (e: React.FormEvent) => {
    e.preventDefault();
    if (!mealName.trim()) return;

    onAddEntry({
      userId: 1,
      meal: mealName.trim(),
      calories: parseInt(calories) || 0,
      protein: parseFloat(protein) || 0,
      carbs: parseFloat(carbs) || 0,
      fats: parseFloat(fats) || 0,
      date: new Date().toISOString().split('T')[0],
    });

    setMealName('');
    setShowAddForm(false);
  };

  // Daily totals
  const todayStr = new Date().toISOString().split('T')[0];
  const todayEntries = dietEntries.filter((e) => e.date.startsWith(todayStr));

  const totalCalories = todayEntries.reduce((acc, e) => acc + e.calories, 0);
  const totalProtein = todayEntries.reduce((acc, e) => acc + e.protein, 0);
  const totalCarbs = todayEntries.reduce((acc, e) => acc + e.carbs, 0);
  const totalFats = todayEntries.reduce((acc, e) => acc + e.fats, 0);

  // Targets
  const targetCalories = 2200;
  const targetProtein = 130;
  const targetCarbs = 240;
  const targetFats = 65;

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header */}
      <div className="pb-5 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-600" />
              <span className="text-xs font-mono uppercase tracking-wider text-slate-500">
                Nutritional Medicine
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1 flex items-center gap-2">
              <Utensils className="w-5 h-5 text-emerald-600" />
              <span>Nutrition & Tissue Recovery Log</span>
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Track macronutrients to support tendon remodeling, reduce muscular inflammation, and accelerate recovery.
            </p>
          </div>

          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold flex items-center gap-2 transition-colors cursor-pointer self-start sm:self-auto shadow-xs"
          >
            <Plus className="w-4 h-4" />
            <span>{showAddForm ? 'Close Entry Form' : 'Log New Meal'}</span>
          </button>
        </div>
      </div>

      {/* Expandable Sections Container */}
      <div className="divide-y divide-slate-200 border-t border-b border-slate-200">
        
        {/* ================================================================ */}
        {/* SECTION 1: TODAY'S MACRONUTRIENT INTAKE */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('macros')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Flame className="w-4 h-4 text-amber-500" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-amber-600 transition-colors">
                  Today's Nutritional Intake
                </h2>
                <span className="text-xs text-slate-500">
                  {totalCalories} / {targetCalories} kcal • {Math.round(totalProtein)}g Protein • {Math.round(totalCarbs)}g Carbs • {Math.round(totalFats)}g Fats
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.macros ? 'Collapse' : 'Expand'}
              </span>
              {openSections.macros ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.macros && (
            <div className="pb-6 pt-2 px-2 text-xs space-y-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 py-2 border-b border-slate-200">
                <div>
                  <div className="flex items-center justify-between text-slate-500 text-[11px] mb-1">
                    <span>Calories</span>
                    <span>{Math.round((totalCalories / targetCalories) * 100)}%</span>
                  </div>
                  <span className="font-mono font-bold text-slate-900 text-base block">{totalCalories} <span className="text-xs text-slate-500 font-normal">/ {targetCalories} kcal</span></span>
                  <div className="w-full h-1.5 bg-slate-100 rounded-full mt-1.5 overflow-hidden">
                    <div className="h-full bg-amber-500 rounded-full" style={{ width: `${Math.min(100, (totalCalories / targetCalories) * 100)}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-slate-500 text-[11px] mb-1">
                    <span>Protein</span>
                    <span>{Math.round((totalProtein / targetProtein) * 100)}%</span>
                  </div>
                  <span className="font-mono font-bold text-emerald-600 text-base block">{Math.round(totalProtein)}g <span className="text-xs text-slate-500 font-normal">/ {targetProtein}g</span></span>
                  <div className="w-full h-1.5 bg-slate-100 rounded-full mt-1.5 overflow-hidden">
                    <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${Math.min(100, (totalProtein / targetProtein) * 100)}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-slate-500 text-[11px] mb-1">
                    <span>Carbohydrates</span>
                    <span>{Math.round((totalCarbs / targetCarbs) * 100)}%</span>
                  </div>
                  <span className="font-mono font-bold text-blue-600 text-base block">{Math.round(totalCarbs)}g <span className="text-xs text-slate-500 font-normal">/ {targetCarbs}g</span></span>
                  <div className="w-full h-1.5 bg-slate-100 rounded-full mt-1.5 overflow-hidden">
                    <div className="h-full bg-blue-600 rounded-full" style={{ width: `${Math.min(100, (totalCarbs / targetCarbs) * 100)}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-slate-500 text-[11px] mb-1">
                    <span>Healthy Fats</span>
                    <span>{Math.round((totalFats / targetFats) * 100)}%</span>
                  </div>
                  <span className="font-mono font-bold text-slate-800 text-base block">{Math.round(totalFats)}g <span className="text-xs text-slate-500 font-normal">/ {targetFats}g</span></span>
                  <div className="w-full h-1.5 bg-slate-100 rounded-full mt-1.5 overflow-hidden">
                    <div className="h-full bg-slate-400 rounded-full" style={{ width: `${Math.min(100, (totalFats / targetFats) * 100)}%` }} />
                  </div>
                </div>
              </div>

              {/* Quick Presets */}
              <div className="pt-2">
                <span className="text-slate-500 text-[11px] block mb-1.5 font-medium">Rehabilitation Quick-Fill Presets:</span>
                <div className="flex flex-wrap gap-2">
                  {presets.map((p, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => applyPreset(p)}
                      className="px-2.5 py-1 rounded bg-white hover:bg-slate-50 text-[11px] text-slate-700 border border-slate-200 transition-colors cursor-pointer shadow-2xs"
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* ADD MEAL FORM (Expandable Form) */}
        {/* ================================================================ */}
        {showAddForm && (
          <div className="py-4 px-2">
            <form onSubmit={handleAddMeal} className="p-4 rounded-lg bg-slate-50 border border-emerald-200 space-y-3.5 text-xs">
              <h3 className="font-semibold text-slate-900">Record Nutrition Entry</h3>
              
              <div>
                <label className="text-slate-700 font-semibold block mb-1">Meal / Food Description</label>
                <input
                  type="text"
                  required
                  value={mealName}
                  onChange={(e) => setMealName(e.target.value)}
                  placeholder="e.g. Grilled Salmon with sweet potato & steamed vegetables"
                  className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <label className="text-slate-600 block mb-1 font-medium">Calories (kcal)</label>
                  <input
                    type="number"
                    min="0"
                    value={calories}
                    onChange={(e) => setCalories(e.target.value)}
                    className="w-full px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-slate-600 block mb-1 font-medium">Protein (g)</label>
                  <input
                    type="number"
                    min="0"
                    value={protein}
                    onChange={(e) => setProtein(e.target.value)}
                    className="w-full px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-slate-600 block mb-1 font-medium">Carbs (g)</label>
                  <input
                    type="number"
                    min="0"
                    value={carbs}
                    onChange={(e) => setCarbs(e.target.value)}
                    className="w-full px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="text-slate-600 block mb-1 font-medium">Fats (g)</label>
                  <input
                    type="number"
                    min="0"
                    value={fats}
                    onChange={(e) => setFats(e.target.value)}
                    className="w-full px-3 py-1.5 bg-white border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setShowAddForm(false)}
                  className="px-3 py-1.5 rounded-lg bg-white text-slate-600 border border-slate-200 text-xs font-semibold cursor-pointer hover:bg-slate-50 hover:text-slate-900"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold transition-colors cursor-pointer shadow-xs"
                >
                  Save Entry
                </button>
              </div>
            </form>
          </div>
        )}

        {/* ================================================================ */}
        {/* SECTION 2: RECORDED MEALS LIST (Clean divider rows) */}
        {/* ================================================================ */}
        <div className="py-2">
          <button
            onClick={() => toggleSection('meals')}
            className="w-full py-4 px-2 flex items-center justify-between text-left group hover:bg-slate-50 transition-colors cursor-pointer select-none"
          >
            <div className="flex items-center gap-3">
              <Utensils className="w-4 h-4 text-emerald-600" />
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-slate-900 group-hover:text-emerald-600 transition-colors">
                  Meal History & Journal
                </h2>
                <span className="text-xs text-slate-500">
                  {dietEntries.length} logged meals
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 hidden sm:inline-block">
                {openSections.meals ? 'Collapse' : 'Expand'}
              </span>
              {openSections.meals ? (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronRight className="w-4 h-4 text-slate-400" />
              )}
            </div>
          </button>

          {openSections.meals && (
            <div className="pb-6 pt-2 px-2 text-xs">
              {dietEntries.length === 0 ? (
                <p className="text-slate-500 text-center py-6">No meals logged yet. Use the Log New Meal button to record intake.</p>
              ) : (
                <div className="divide-y divide-slate-200">
                  {dietEntries.map((entry) => (
                    <div
                      key={entry.id}
                      className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 hover:bg-slate-50 px-1 rounded transition-colors"
                    >
                      <div className="space-y-0.5">
                        <span className="font-bold text-slate-900 text-xs sm:text-sm">{entry.meal}</span>
                        <div className="flex items-center gap-3 text-[11px] text-slate-500">
                          <span>{entry.date}</span>
                          <span>•</span>
                          <span className="font-mono text-slate-700 font-medium">{entry.calories} kcal</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-4 shrink-0 self-end sm:self-center">
                        <div className="text-right text-[11px] font-mono text-slate-600">
                          P: <strong className="text-emerald-600">{entry.protein}g</strong> • C: <strong className="text-blue-600">{entry.carbs}g</strong> • F: <strong className="text-slate-700">{entry.fats}g</strong>
                        </div>
                        <button
                          onClick={() => onDeleteEntry(entry.id)}
                          className="p-1.5 rounded text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors cursor-pointer"
                          title="Delete Entry"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
