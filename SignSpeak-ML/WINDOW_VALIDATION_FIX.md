# FINAL FIX: Window Validation Before Classification

## Problem Solved

**Root Cause:** Sliding Window klasyfikował OKNA zawierające MIX starych ramek z gestem + nowe puste ramki.

Przykład:
```
Buffer: [old_gesture_frames 1-40, empty_frames 41-100]
Window [frames 10-70]: 30 starych ramek z gestem + 30 pustych
→ Klasyfikator widzi fragmenty gestu → "THANK YOU" 98%! ❌
```

## Implementacja

### Dodano w `SlidingWindowDetector.py` (linia ~188):

```python
# CHECK: Skip classification if window has too many empty frames
# This prevents classifying windows with MIX of old gesture frames + new empty frames
window_magnitudes = np.linalg.norm(window_np, axis=1)
non_empty_count = np.sum(window_magnitudes > 0.01)
empty_ratio = 1.0 - (non_empty_count / len(window_magnitudes))

# If window is mostly empty (>50% empty frames), skip classification
if empty_ratio > 0.5:
    print(f"[SKIP WINDOW] Too many empty frames: {empty_ratio:.1%} empty ({non_empty_count}/{len(window_magnitudes)} non-empty)")
    # Reset state to prevent detecting old gestures
    self.last_predicted_word = None
    self.consecutive_count = 0
    return None
```

## Jak to działa

### 1. Sprawdzenie każdej ramki w window
```python
window_magnitudes = np.linalg.norm(window_np, axis=1)  # Magnitude każdej ramki
non_empty_count = np.sum(window_magnitudes > 0.01)     # Ile niepustych
```

### 2. Obliczenie ratio pustych ramek
```python
empty_ratio = 1.0 - (non_empty_count / len(window_magnitudes))
```

### 3. Skip jeśli > 50% pustych
```python
if empty_ratio > 0.5:
    # DON'T classify this window!
    return None
```

## Przykłady

### Przykład 1: Window z głównie starymi ramkami
```
Window: [20 starych ramek z gestem, 40 pustych ramek]
non_empty_count: 20
empty_ratio: 66.7%
→ SKIP! ✅
```

### Przykład 2: Window z aktywnym gestem
```
Window: [50 ramek z gestem, 10 pustych ramek]
non_empty_count: 50
empty_ratio: 16.7%
→ CLASSIFY ✅
```

### Przykład 3: Window z samymi pustymi ramkami
```
Window: [0 ramek z gestem, 60 pustych ramek]
non_empty_count: 0
empty_ratio: 100%
→ SKIP! ✅
```

## Dlaczego wcześniejsze rozwiązania nie działały

### ❌ Rozwiązanie 1: Detekcja pojedynczej pustej ramki
```python
if frame_magnitude < 0.01:
    self._consecutive_empty_frames += 1
```
**Problem:** Sprawdza tylko NOWĄ ramkę, nie cały window!

### ❌ Rozwiązanie 2: Czyszczenie bufora po 60 pustych ramkach
```python
if self._consecutive_empty_frames > self.window_size:
    self.frame_buffer.clear()
```
**Problem:** Buffer nadal zawiera stare ramki w pierwszych 60 nowych pustych ramkach!

### ✅ Rozwiązanie 3: Sprawdzenie całego window przed klasyfikacją
```python
# Count non-empty frames IN THE WINDOW
if empty_ratio > 0.5:
    return None  # Skip classification
```
**Działa:** Zapobiega klasyfikacji okien z głównie pustymi ramkami!

## Logs do obserwacji

Konsola API będzie pokazywać:
```
[SKIP WINDOW] Too many empty frames: 73.3% empty (16/60 non-empty)
[SKIP WINDOW] Too many empty frames: 83.3% empty (10/60 non-empty)
[SKIP WINDOW] Too many empty frames: 91.7% empty (5/60 non-empty)
[SKIP WINDOW] Too many empty frames: 100.0% empty (0/60 non-empty)
```

## Test

### Uruchom API i realtime test:
```powershell
# Terminal 1
uvicorn app.main:app --reload

# Terminal 2
python eval\realtime_test.py
```

### Oczekiwany rezultat:

**Scenariusz 1: Start z rękami poza kamerą**
- ❌ PRZED: Wykrywa "THANK YOU" 98% od razu
- ✅ PO: Brak detekcji, logi `[SKIP WINDOW]`

**Scenariusz 2: Gest → Schowaj ręce**
- ❌ PRZED: Wykrywa "THANK YOU" w kółko
- ✅ PO: 1 detekcja, potem `[SKIP WINDOW]`

**Scenariusz 3: Gest → Pauza → Nowy gest**
- ❌ PRZED: Wykrywa stary gest podczas pauzy
- ✅ PO: Czyste oddzielenie słów

## Configuration

### Threshold dla pustych ramek (dostosuj jeśli potrzeba):

**Bardziej restrykcyjny (wymaga 70% aktywnych ramek):**
```python
if empty_ratio > 0.3:  # Zmień z 0.5 na 0.3
```

**Mniej restrykcyjny (wymaga 30% aktywnych ramek):**
```python
if empty_ratio > 0.7:  # Zmień z 0.5 na 0.7
```

**Aktualne (wymaga 50% aktywnych ramek):**
```python
if empty_ratio > 0.5:  # Default - dobry balans
```

## Files Changed

1. `app/model_logic/segmentation/SlidingWindowDetector.py`
   - Dodano window validation (linie ~188-199)
   - Sprawdza ratio pustych ramek przed klasyfikacją
   - Skip jeśli > 50% window to puste ramki

## Previous Fixes (already in place)

1. ✅ Detekcja pojedynczej pustej ramki (linia ~141)
2. ✅ Reset po 60 pustych ramkach (linia ~151)
3. ✅ Reset po emitowaniu słowa (linia ~242)

## All Together

Kombinacja wszystkich 3 fixes:
1. **Single frame detection** → Śledzi puste ramki
2. **Buffer clearing** → Czyści po długiej ciszy
3. **Window validation** ⭐ → Zapobiega klasyfikacji mieszanych okien

**Tylko fix #3 naprawia ghost detection od razu przy starcie!**

---

**Status:** ✅ ZAIMPLEMENTOWANE  
**Tested:** Ready for testing  
**Priority:** CRITICAL - Naprawia główny bug  
**Date:** 2025-01-30

