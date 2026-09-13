export interface DietEntry {
  id: number;
  user_id: number;
  meal: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
  date: string | null;
}

export interface DietSummary {
  total_calories: number;
  total_protein: number;
  total_carbs: number;
  total_fats: number;
  entries: DietEntry[];
}
