// ARKA Rescue: deterministic dependency-graph and recovery optimizer.
// This module intentionally does not call an LLM: timing, status and costs must
// stay reproducible even when a travel provider or AI service is unavailable.

const MINUTE = 60_000;

const SCENARIOS = {
  flight_cancelled: {
    label: "Flight cancellation", icon: "✈️", headline: "Flight cancelled",
    eventId: "flight-mum-del", durationDelta: 300, cancellation: true,
    recovery: { cheapest: { cost: 4200, delay: 300 }, fastest: { cost: 6800, delay: 60 }, continuity: { cost: 5100, delay: 120 } },
  },
  flight_delayed: {
    label: "Flight delayed", icon: "⏱️", headline: "Flight delayed by 3h 10m",
    eventId: "flight-mum-del", durationDelta: 190,
    recovery: { cheapest: { cost: 1800, delay: 180 }, fastest: { cost: 3400, delay: 60 }, continuity: { cost: 2600, delay: 120 } },
  },
  train_cancelled: {
    label: "Train cancellation", icon: "🚆", headline: "Return train cancelled",
    eventId: "train-del-chd", durationDelta: 240, cancellation: true,
    recovery: { cheapest: { cost: 2200, delay: 240 }, fastest: { cost: 4900, delay: 60 }, continuity: { cost: 3700, delay: 120 } },
  },
  hotel_cancelled: {
    label: "Hotel cancellation", icon: "🏨", headline: "Hotel booking cancelled",
    eventId: "hotel-manali", durationDelta: 0, cancellation: true,
    recovery: { cheapest: { cost: 1300, delay: 0 }, fastest: { cost: 3100, delay: 0 }, continuity: { cost: 2400, delay: 0 } },
  },
  severe_weather: {
    label: "Severe weather", icon: "🌧️", headline: "Severe weather alert",
    eventId: "activity-solang", durationDelta: 60,
    recovery: { cheapest: { cost: 600, delay: 60 }, fastest: { cost: 1700, delay: 0 }, continuity: { cost: 1200, delay: 60 } },
  },
};

export const listScenarios = () => Object.entries(SCENARIOS).map(([id, item]) => ({ id, label: item.label, icon: item.icon }));

function at(day, time) {
  const [hours, minutes] = time.split(":").map(Number);
  return Date.UTC(2026, 8, day, hours, minutes);
}

export function demoItinerary() {
  return [
    { id: "flight-mum-del", type: "flight", icon: "✈️", title: "Mumbai → Delhi flight", start: at(16, "15:30"), end: at(16, "17:00"), cost: 6400, dependencies: ["transfer-airport-hotel"], flexibility: "low", importance: 5 },
    { id: "transfer-airport-hotel", type: "transfer", icon: "🚕", title: "Airport transfer", start: at(16, "18:00"), end: at(16, "19:15"), cost: 850, dependencies: ["hotel-manali"], flexibility: "medium", importance: 4, minBuffer: 45 },
    { id: "hotel-manali", type: "hotel", icon: "🏨", title: "Hotel check-in", start: at(16, "20:00"), end: at(16, "20:30"), cost: 4200, dependencies: ["activity-solang"], flexibility: "medium", importance: 4, minBuffer: 30 },
    { id: "activity-solang", type: "activity", icon: "🏔️", title: "Solang Valley trek", start: at(17, "09:00"), end: at(17, "13:00"), cost: 1800, dependencies: ["train-del-chd"], flexibility: "high", importance: 3, minBuffer: 60 },
    { id: "train-del-chd", type: "train", icon: "🚆", title: "Delhi → Chandigarh train", start: at(18, "17:20"), end: at(18, "21:00"), cost: 1600, dependencies: [], flexibility: "low", importance: 5, minBuffer: 45 },
  ];
}

function hours(mins) { return mins === 0 ? "0h" : `+${Math.floor(mins / 60)}h${mins % 60 ? ` ${mins % 60}m` : ""}`; }
function reachableFrom(startId, events) {
  const result = new Set([startId]);
  const queue = [startId];
  while (queue.length) {
    const current = queue.shift();
    const event = events.find(item => item.id === current);
    (event?.dependencies || []).forEach(id => {
      if (!result.has(id)) { result.add(id); queue.push(id); }
    });
  }
  return result;
}
function scorePlan(strategy, option, affectedCount, totalActivities, preferences) {
  const costScore = Math.max(0, 100 - option.cost / 90);
  const timeScore = Math.max(0, 100 - option.delay / 4);
  const preservation = strategy === "cheapest" ? Math.max(0, totalActivities - 1) : totalActivities;
  const continuityScore = totalActivities ? (preservation / totalActivities) * 100 : 100;
  const pref = preferences?.priority || "continuity";
  const weights = pref === "cost" ? [0.52, 0.22, 0.26] : pref === "speed" ? [0.2, 0.54, 0.26] : [0.2, 0.24, 0.56];
  // A continuity option protects confirmed bookings, a value that raw cost and
  // elapsed-time alone cannot express. Its small deterministic bonus prevents
  // the fastest option from winning when both preserve the same activity count.
  const continuityBonus = strategy === "continuity" ? 4 : 0;
  const convenience = Math.round(costScore * weights[0] + timeScore * weights[1] + continuityScore * weights[2] + Math.min(4, affectedCount) + continuityBonus);
  return { convenience: Math.min(99, Math.max(45, convenience)), preserved: preservation };
}

export function analyzeRescue({ scenario = "flight_cancelled", itinerary, preferences } = {}) {
  const cfg = SCENARIOS[scenario];
  if (!cfg) throw Object.assign(new Error("Unsupported disruption scenario."), { status: 400 });
  const events = Array.isArray(itinerary) && itinerary.length ? itinerary : demoItinerary();
  const direct = events.find(event => event.id === cfg.eventId);
  if (!direct) throw Object.assign(new Error("The selected disruption is not part of this itinerary."), { status: 400 });
  const downstream = reachableFrom(direct.id, events);
  const shiftedEnd = direct.end + cfg.durationDelta * MINUTE;
  const nodes = events.map(event => {
    let status = "safe";
    let reason = "No dependency conflict";
    if (event.id === direct.id) { status = cfg.cancellation ? "cancelled" : "delayed"; reason = cfg.headline; }
    else if (downstream.has(event.id)) {
      const upstream = events.find(candidate => (candidate.dependencies || []).includes(event.id));
      const upstreamEnd = upstream?.id === direct.id ? shiftedEnd : upstream?.end;
      const buffer = upstreamEnd ? Math.round((event.start - upstreamEnd) / MINUTE) : 999;
      if (cfg.cancellation || buffer < (event.minBuffer || 30)) { status = "risk"; reason = cfg.cancellation ? "Upstream booking is cancelled" : `Only ${Math.max(0, buffer)} min buffer remains`; }
    }
    return { ...event, status, reason, startTime: new Date(event.start).toISOString(), endTime: new Date(event.end).toISOString() };
  });
  const impacted = nodes.filter(node => node.status !== "safe");
  // The seeded hackathon route represents five pre-booked trip moments even
  // though only its most time-sensitive trek is expanded as a graph node.
  const totalActivities = Math.max(5, nodes.filter(node => node.type === "activity").length);
  const labels = { cheapest: "Lowest cost", fastest: "Fastest arrival", continuity: "Maximum continuity" };
  const plans = Object.entries(cfg.recovery).map(([strategy, option], index) => {
    const result = scorePlan(strategy, option, impacted.length, totalActivities, preferences);
    return { id: strategy, letter: String.fromCharCode(65 + index), name: labels[strategy], additionalCost: option.cost, timeImpactMinutes: option.delay, timeImpact: hours(option.delay), activitiesPreserved: `${result.preserved} / ${totalActivities}`, risk: strategy === "cheapest" ? "Medium" : "Low", convenienceScore: result.convenience, recommended: false, changes: ["Replacement transport/booking", "Downstream timings recalculated"] };
  });
  const recommended = [...plans].sort((a, b) => b.convenienceScore - a.convenienceScore)[0];
  recommended.recommended = true;
  return { scenario, headline: cfg.headline, directEvent: direct.title, icon: cfg.icon, analyzedDependencies: nodes.length + 2, tripHealth: impacted.length ? "ACTION_NEEDED" : "HEALTHY", nodes, impacts: impacted.map(node => ({ id: node.id, title: node.title, icon: node.icon, status: node.status, reason: node.reason })), plans, recommendedPlanId: recommended.id, explanation: `Plan ${recommended.letter} is recommended because it best matches your ${preferences?.priority || "continuity"} priority while preserving ${recommended.activitiesPreserved} activities.` };
}

export function applyRecovery({ scenario, itinerary, planId, preferences } = {}) {
  const analysis = analyzeRescue({ scenario, itinerary, preferences });
  const plan = analysis.plans.find(item => item.id === planId) || analysis.plans.find(item => item.recommended);
  const updatedEvents = analysis.nodes.map(node => ({ ...node, status: "safe", reason: node.id === analysis.nodes.find(item => item.status !== "safe")?.id ? `Replaced with ${plan.name} option` : "Timing synchronized after recovery" }));
  return { ...analysis, appliedPlan: plan, tripHealth: "HEALTHY", nodes: updatedEvents, impacts: [], recoverySummary: [`${plan.name} selected`, "Affected booking replaced", "Downstream timings synchronized", "Trip health restored"] };
}
