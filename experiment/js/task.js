
// Data
let start_task_time = 0;
let subjectData = {};
let messageData = '';

// Collect prolific id
function handle_prolific() {
  subjectData['prolific_id'] = getEl('prolific_id_text').value;
  hideAndShowNext('prolific_id', 'instruction', 'block');
}

// Grid world task
function grid_done() {
  hide('task-info');
  getEl("task-content").style.marginTop = "40px";
  hide('task-grid');
  showNext('task-composer', 'block');
}

// Message composer
function handle_submit() {
  const composerText = document.getElementById('composer-text').value;

  if (composerText === "") {
    alert("Please type your message. You will be bonused for the quality of your message.");
    return;
  }

  subjectData['message'] = composerText;
  hideAndShowNext('task', 'debrief', 'block');
}

// Bebrief
function enableDoneButton() {
  const doneButton = getEl('done-btn');
  if (isFilled('postquiz')) {
    doneButton.disabled = false;
  } else {
    doneButton.disabled = true;
  }
}
// Add event listeners to form inputs
const postquizForm = getEl('postquiz');
const inputs = postquizForm.elements;
for (let i = 0; i < inputs.length; i++) {
  if (inputs[i].type !== "submit") {
    inputs[i].addEventListener('input', enableDoneButton);
  }
}

function is_done(complete_code) {
  let inputs = getEl('postquiz').elements;
  Object.keys(inputs).forEach(id => subjectData[inputs[id].name] = inputs[id].value);

  // Get data
  subjectData['feedback'] = removeSpecial(subjectData['feedback']);
  subjectData['condition'] = COND;
  subjectData['total_points'] = POINTS;

  const end_time = new Date();

  let clientData = {};
  clientData.subject = subjectData;
  clientData.subject.date = formatDates(end_time, 'date');
  clientData.subject.time = formatDates(end_time, 'time');
  clientData.subject.task_duration = end_time - start_task_time;
  clientData.subject.token = token;

  clientData.actions = actionData;
  clientData.events = eventData;


  // Show completion
  hide("debrief")
  showNext("completed")
  getEl('completion-code').append(document.createTextNode(complete_code));


  // Save data
  console.log(clientData)
  // download(JSON.stringify(clientData), 'data.txt', '"text/csv"');
}
